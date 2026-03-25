from __future__ import annotations


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def _consume_fenced_code(markdown: str, start: int) -> tuple[str, int] | None:
    if start > 0 and markdown[start - 1] != "\n":
        return None

    index = start
    spaces = 0
    while index < len(markdown) and markdown[index] == " " and spaces < 4:
        index += 1
        spaces += 1

    if spaces > 3 or index >= len(markdown) or markdown[index] not in {"`", "~"}:
        return None

    fence_char = markdown[index]
    fence_end = index
    while fence_end < len(markdown) and markdown[fence_end] == fence_char:
        fence_end += 1

    fence_length = fence_end - index
    if fence_length < 3:
        return None

    line_end = markdown.find("\n", fence_end)
    if line_end == -1:
        return markdown[start:], len(markdown)

    line_start = line_end + 1
    while line_start < len(markdown):
        line_end = markdown.find("\n", line_start)
        if line_end == -1:
            line_end = len(markdown)

        index = line_start
        spaces = 0
        while index < line_end and markdown[index] == " " and spaces < 4:
            index += 1
            spaces += 1

        fence_end = index
        while fence_end < line_end and markdown[fence_end] == fence_char:
            fence_end += 1

        if (
            fence_end - index >= fence_length
            and markdown[fence_end:line_end].strip() == ""
        ):
            block_end = line_end + (1 if line_end < len(markdown) else 0)
            return markdown[start:block_end], block_end

        line_start = line_end + 1

    return markdown[start:], len(markdown)


def _consume_code_span(markdown: str, start: int) -> tuple[str, int]:
    index = start
    while index < len(markdown) and markdown[index] == "`":
        index += 1
    ticks = markdown[start:index]
    end = markdown.find(ticks, index)
    if end == -1:
        return markdown[start:index], index
    end += len(ticks)
    return markdown[start:end], end


def _consume_math(markdown: str, start: int) -> tuple[str, int] | None:
    delimiters = (
        ("$$", "$$"),
        (r"\[", r"\]"),
        (r"\(", r"\)"),
        ("$", "$"),
    )
    for opening, closing in delimiters:
        if not markdown.startswith(opening, start):
            continue
        if opening.startswith("$") and _is_escaped(markdown, start):
            return None
        if opening == "$" and markdown.startswith("$$", start):
            continue
        index = start + len(opening)
        while index < len(markdown):
            if markdown.startswith(closing, index) and not _is_escaped(markdown, index):
                inner = markdown[start + len(opening) : index]
                return opening + _escape_underscores(inner) + closing, index + len(closing)
            index += 1
        return None
    return None


def preprocess(markdown: str) -> str:
    output: list[str] = []
    index = 0
    while index < len(markdown):
        fenced = _consume_fenced_code(markdown, index)
        if fenced is not None:
            block, index = fenced
            output.append(block)
            continue

        if markdown[index] == "`":
            span, index = _consume_code_span(markdown, index)
            output.append(span)
            continue

        math = _consume_math(markdown, index)
        if math is not None:
            block, index = math
            output.append(block)
            continue

        output.append(markdown[index])
        index += 1

    return "".join(output)
