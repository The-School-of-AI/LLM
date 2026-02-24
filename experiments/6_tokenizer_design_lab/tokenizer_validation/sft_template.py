from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChatTemplate:
    bos_token: str = "<|begin_of_text|>"
    eos_token: str = "<|end_of_text|>"
    system_token: str = "<|system|>"
    user_token: str = "<|user|>"
    assistant_token: str = "<|assistant|>"


def render_chat(*, system: str | None, user: str, assistant: str) -> str:
    tpl = ChatTemplate()
    parts: list[str] = [tpl.bos_token]

    if system is not None and system.strip():
        parts.extend([tpl.system_token, "\n", system.strip(), "\n"])

    parts.extend([tpl.user_token, "\n", user.strip(), "\n"])
    parts.extend([tpl.assistant_token, "\n", assistant.strip(), "\n", tpl.eos_token])
    return "".join(parts)


def try_split_prompt_answer(line: str) -> tuple[str, str] | None:
    s = line.strip()
    if not s:
        return None

    for delim in ("\t", "|||", "\u241f"):
        if delim in s:
            left, right = s.split(delim, 1)
            left = left.strip()
            right = right.strip()
            if left and right:
                return left, right

    # Heuristic: if there is a single question mark, treat everything through it as prompt.
    if "?" in s:
        idx = s.find("?")
        prompt = s[: idx + 1].strip()
        answer = s[idx + 1 :].strip()
        if prompt and answer:
            return prompt, answer

    return None
