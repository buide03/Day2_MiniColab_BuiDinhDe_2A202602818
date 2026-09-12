"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""

class ChatbotBaseline:
    """Single-turn chatbot that does not use the local tools."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def query(self, user_input: str) -> Dict[str, Any]:
        """Ask Gemini when configured, otherwise return the offline baseline."""
        if not self.api_key:
            return {
                "status": "success",
                "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
                "tool_calls": []
            }

        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError(
                "Gemini API cần package google-generativeai. "
                "Hãy cài dependencies trong starter-code/requirements.txt."
            ) from exc

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            "Bạn là chatbot tư vấn du lịch. "
            "Hãy trả lời câu hỏi sau, không dùng tool hay internet:\n"
            f"{user_input}"
        )
        return {
            "status": "success",
            "answer": response.text,
            "tool_calls": []
        }

class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop"""
    def __init__(self, max_iterations: int = 5, api_key: Optional[str] = None):
        if max_iterations < 1:
            raise ValueError("max_iterations phải lớn hơn hoặc bằng 1")
        self.max_iterations = max_iterations
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.trace: List[Dict[str, Any]] = []

    def run(self, user_input: str) -> Dict[str, Any]:
        if self.api_key:
            return self._run_with_llm(user_input)
        return self._run_deterministic(user_input)

    def _run_deterministic(self, user_input: str) -> Dict[str, Any]:
        """Run a deterministic Thought-Action-Observation loop.

        This offline path keeps the lab runnable without an API key.
        """
        self.trace = []
        observations: Dict[str, Any] = {}
        requested_tool_count = sum(
            (
                self._contains_flight_request(user_input.lower()),
                self._contains_weather_request(user_input.lower()),
            )
        )

        for iteration in range(1, self.max_iterations + 1):
            action = self._plan_action(user_input, observations)
            trace_entry: Dict[str, Any] = {
                "iteration": iteration,
                "thought": self._thought_for(action, observations),
            }

            if action is None:
                answer = self._build_final_answer(user_input, observations)
                trace_entry["final_answer"] = answer
                self.trace.append(trace_entry)
                return {
                    "status": "completed",
                    "iterations": iteration,
                    "answer": answer,
                    "trace": self.trace,
                }

            trace_entry["action"] = action
            observation = self._execute_action(action)
            trace_entry["observation"] = observation
            self.trace.append(trace_entry)
            observations[action["name"]] = observation

            if isinstance(observation, dict) and "error" in observation:
                answer = f"Không thể hoàn thành yêu cầu: {observation['error']}."
                trace_entry["final_answer"] = answer
                return {
                    "status": "completed",
                    "iterations": iteration,
                    "answer": answer,
                    "trace": self.trace,
                }

            # A one-tool request can be answered in the same iteration as its
            # observation; multi-tool requests reserve a final iteration for
            # composing the combined response.
            if requested_tool_count == 1:
                answer = self._build_final_answer(user_input, observations)
                trace_entry["final_answer"] = answer
                return {
                    "status": "completed",
                    "iterations": iteration,
                    "answer": answer,
                    "trace": self.trace,
                }

        return {
            "status": "max_iterations_reached",
            "iterations": self.max_iterations,
            "answer": "Không thể hoàn thành trong số bước tối đa.",
            "trace": self.trace,
        }

    def _run_with_llm(self, user_input: str) -> Dict[str, Any]:
        """Run the actual Thought-Action-Observation loop with Gemini."""
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError(
                "ReActAgent cần package google-generativeai. "
                "Hãy cài dependencies trong starter-code/requirements.txt."
            ) from exc

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = self._build_llm_prompt(user_input)
        self.trace = []

        for iteration in range(1, self.max_iterations + 1):
            response = model.generate_content(prompt)
            raw_response = response.text.strip()
            parsed = self._parse_llm_response(raw_response)

            if parsed is None:
                observation = "Invalid JSON format. Return only valid JSON."
                self.trace.append({
                    "iteration": iteration,
                    "raw_response": raw_response,
                    "observation": observation,
                })
                prompt = self._append_observation(prompt, observation)
                continue

            if parsed.get("type") == "final_answer":
                answer = parsed.get("answer")
                if not isinstance(answer, str) or not answer.strip():
                    observation = "Invalid final_answer. The answer must be a non-empty string."
                    self.trace.append({
                        "iteration": iteration,
                        "raw_response": raw_response,
                        "observation": observation,
                    })
                    prompt = self._append_observation(prompt, observation)
                    continue

                self.trace.append({
                    "iteration": iteration,
                    "thought": parsed.get("thought", ""),
                    "final_answer": answer,
                })
                return {
                    "status": "completed",
                    "iterations": iteration,
                    "answer": answer,
                    "trace": self.trace,
                }

            if parsed.get("type") != "action":
                observation = "Invalid response type. Use 'action' or 'final_answer'."
                self.trace.append({
                    "iteration": iteration,
                    "raw_response": raw_response,
                    "observation": observation,
                })
                prompt = self._append_observation(prompt, observation)
                continue

            action = {
                "name": parsed.get("name"),
                "args": parsed.get("args", {}),
            }
            observation = self._execute_action(action)
            self.trace.append({
                "iteration": iteration,
                "thought": parsed.get("thought", ""),
                "action": action,
                "observation": observation,
            })
            prompt = self._append_observation(prompt, observation)

        return {
            "status": "max_iterations_reached",
            "iterations": self.max_iterations,
            "answer": "Không thể hoàn thành trong số bước tối đa.",
            "trace": self.trace,
        }

    @staticmethod
    def _build_llm_prompt(user_input: str) -> str:
        tools = json.dumps(TOOL_DEFINITIONS, ensure_ascii=False, indent=2)
        return (
            SYSTEM_PROMPT.format(tools=tools)
            + "\nChỉ trả về JSON hợp lệ, không markdown và không bao quanh bằng ```.\n"
            + "Dùng một trong hai dạng sau:\n"
            + '{"type":"action","thought":"...","name":"tool_name","args":{...}}\n'
            + '{"type":"final_answer","thought":"...","answer":"..."}\n'
            + f"\nUser input:\n{user_input}"
        )

    @staticmethod
    def _append_observation(prompt: str, observation: Any) -> str:
        serialized = json.dumps(observation, ensure_ascii=False)
        return f"{prompt}\nObservation: {serialized}\nChọn action tiếp theo hoặc final_answer."

    @staticmethod
    def _parse_llm_response(raw_response: str) -> Optional[Dict[str, Any]]:
        candidate = raw_response.strip()
        if candidate.startswith("```") and candidate.endswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate).strip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _normalise_tool_name(name: str) -> str:
        return name.strip().lower()

    def _plan_action(
        self, user_input: str, observations: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Select the next action from the request and completed observations."""
        text = user_input.lower()
        if "get_flight_info" not in observations and self._contains_flight_request(text):
            origin, destination = self._extract_route(text)
            if origin and destination:
                return {
                    "name": "get_flight_info",
                    "args": {
                        "origin": origin,
                        "destination": destination,
                        "max_price": self._extract_max_price(text),
                    },
                }

        if "get_weather_forecast" not in observations and self._contains_weather_request(text):
            city_code = self._extract_city_code(text)
            if city_code:
                return {
                    "name": "get_weather_forecast",
                    "args": {"city_code": city_code},
                }

        return None

    @staticmethod
    def _contains_flight_request(text: str) -> bool:
        return any(word in text for word in ("chuyến bay", "vé máy bay", "tìm vé", "flight"))

    @staticmethod
    def _contains_weather_request(text: str) -> bool:
        return any(word in text for word in ("thời tiết", "nhiệt độ", "mặc gì", "weather"))

    @staticmethod
    def _extract_route(text: str) -> tuple[Optional[str], Optional[str]]:
        match = re.search(
            r"\b(?:từ|from)\s+([a-z]{3})\s+(?:đi|đến|to)\s+([a-z]{3})\b",
            text,
        )
        if not match:
            match = re.search(r"\b([a-z]{3})\s+(?:đến|đi|to)\s+([a-z]{3})\b", text)
        if not match:
            return None, None
        return match.group(1).upper(), match.group(2).upper()

    @staticmethod
    def _extract_max_price(text: str) -> int:
        match = re.search(r"(?:dưới|tối đa|less than|under)\s*([\d,.]+)\s*(triệu|tr)\b", text)
        if match:
            return int(float(match.group(1).replace(",", ".")) * 1_000_000)

        match = re.search(r"(?:dưới|tối đa|under)\s*([\d,]+)", text)
        if match:
            return int(match.group(1).replace(",", ""))
        return 5_000_000

    @staticmethod
    def _extract_city_code(text: str) -> Optional[str]:
        codes = ("SGN", "HAN", "DAD")
        for code in codes:
            if re.search(rf"\b{code.lower()}\b", text):
                return code
        city_names = {
            "thành phố hồ chí minh": "SGN",
            "tp. hồ chí minh": "SGN",
            "sài gòn": "SGN",
            "hà nội": "HAN",
            "đà nẵng": "DAD",
        }
        for city, code in city_names.items():
            if city in text:
                return code
        return None

    def _thought_for(
        self, action: Optional[Dict[str, Any]], observations: Dict[str, Any]
    ) -> str:
        if action is None:
            return "Đã có đủ dữ liệu hoặc đây là câu hỏi không cần tool; tạo câu trả lời cuối."
        if action["name"] == "get_flight_info":
            return "Cần tra cứu các chuyến bay phù hợp với tuyến đường và ngân sách."
        return "Cần tra cứu thời tiết để đưa ra thông tin và gợi ý trang phục."

    def _execute_action(self, action: Dict[str, Any]) -> Any:
        if not isinstance(action.get("name"), str):
            return {"error": "Invalid action: name must be a string"}
        tool_name = self._normalise_tool_name(action["name"])
        tool = TOOL_MAP.get(tool_name)
        if tool is None:
            return {"error": f"Unknown tool: {tool_name}"}

        args = action.get("args", {})
        if not isinstance(args, dict):
            return {"error": f"Invalid arguments for {tool_name}: args must be an object"}
        try:
            return tool(**args)
        except (TypeError, ValueError) as exc:
            return {"error": f"Invalid arguments for {tool_name}: {exc}"}

    @staticmethod
    def _build_final_answer(user_input: str, observations: Dict[str, Any]) -> str:
        parts: List[str] = []
        flights = observations.get("get_flight_info")
        if isinstance(flights, list):
            if flights:
                flight_text = ", ".join(
                    f"{flight['flight_number']} ({flight['airline']}, "
                    f"{flight['price_vnd']:,} VND, {flight['departure_time']})"
                    for flight in flights
                )
                parts.append(f"Các chuyến bay phù hợp: {flight_text}.")
            else:
                parts.append("Không tìm thấy chuyến bay phù hợp.")

        weather = observations.get("get_weather_forecast")
        if isinstance(weather, dict) and "city" in weather:
            parts.append(
                f"Thời tiết tại {weather['city']}: {weather['temperature_c']}°C, "
                f"{weather['condition']}. Gợi ý: {weather['recommendation']}"
            )

        if parts:
            return " ".join(parts)
        return f"Tôi chưa có tool phù hợp để xử lý yêu cầu: {user_input}"

def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"
    
    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))
    
    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result)
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()