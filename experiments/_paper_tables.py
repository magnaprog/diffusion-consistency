"""Extract LaTeX table rows for comparison with the study records.

A number found elsewhere in the manuscript does not establish that the
corresponding table cell is correct. These helpers locate a labelled table,
remove comments before flattening whitespace, and return cleaned cell text
so callers can compare the expected rows. The source records that earlier
checks in both workshop papers accepted deliberately corrupted table rows;
that failure motivated explicit row and cell checks.

This is a parser for the supported LaTeX patterns, not a renderer.
assert_rendered checks substring presence in supplied PDF text. Callers must
delimit the relevant table and check complete rows when location matters.

Usage:
    rows = tabular_rows(tex, "tab:arms")
    assert rows["Trained"] == ["11", "0.306 +- 0.029", "0.201 +- 0.016"]
"""
import re


def strip_comments(tex):
    """Remove LaTeX line comments before normalizing whitespace.

    Line breaks identify where a comment ends. Removing them first could
    make a commented label or table definition look like active source and
    cause the verifier to inspect numbers absent from the PDF.

    An unescaped percent sign starts a comment. Count immediately preceding
    backslashes to distinguish an escaped percent sign from a comment after
    a LaTeX line-break command.
    """
    kept = []
    for line in tex.split("\n"):
        index = 0
        while True:
            index = line.find("%", index)
            if index == -1:
                break
            backslashes = 0
            probe = index - 1
            while probe >= 0 and line[probe] == "\\":
                backslashes += 1
                probe -= 1
            if backslashes % 2 == 0:
                line = line[:index]
                break
            index += 1
        kept.append(line)
    return "\n".join(kept)

_STRIP = (
    (r"\\textbf\{([^{}]*)\}", r"\1"),
    (r"\\mathbf\{([^{}]*)\}", r"\1"),
    (r"\\emph\{([^{}]*)\}", r"\1"),
    (r"\\multicolumn\{\d+\}\{[^{}]*\}\{([^{}]*)\}", r"\1"),
    (r"\\pm", "+-"),
    (r"\\times", "x"),
    (r"\\%", "%"),
    (r"\\,", " "),
    (r"\\ ", " "),
    (r"\$", ""),
    (r"\{,\}", ""),
    (r"\\[a-zA-Z]+", ""),
    (r"[{}]", ""),
)


def clean_cell(text):
    """Normalize one source cell using the listed LaTeX substitutions.

    Return text suitable for the caller's comparisons. The substitutions
    remove selected formatting commands and collapse whitespace; they do
    not evaluate arbitrary LaTeX or establish what the PDF displays.
    """
    for pattern, replacement in _STRIP:
        text = re.sub(pattern, replacement, text)
    return " ".join(text.split())


def float_block(tex, label):
    """One non-nested table environment carrying a unique ``label``."""
    clean = strip_comments(tex)
    marker = "\\label{" + label + "}"
    occurrences = clean.count(marker)
    if occurrences == 0:
        raise KeyError(f"no float labelled {label}")
    if occurrences > 1:
        raise KeyError(f"{occurrences} floats labelled {label}; cannot bind")
    matches = []
    pattern = re.compile(
        r"\\(?P<kind>begin|end)\{(?P<env>table\*?)\}"
        r"(?:\[[^]]*\])?")
    start = None
    environment = None
    for token in pattern.finditer(clean):
        if token.group("kind") == "begin":
            if start is not None:
                raise KeyError("nested table environments cannot be bound")
            start = token.start()
            environment = token.group("env")
            continue
        if start is None:
            raise KeyError(f"unmatched \\end{{{token.group('env')}}}")
        if token.group("env") != environment:
            raise KeyError(
                f"table environment closes as {token.group('env')}, "
                f"not {environment}")
        block = clean[start:token.end()]
        if marker in block:
            matches.append(block)
        start = None
        environment = None
    if start is not None:
        raise KeyError(f"unterminated \\begin{{{environment}}}")
    if len(matches) != 1:
        raise KeyError(f"label {label} belongs to {len(matches)} table environments")
    block = matches[0]
    if re.search(r"\\if[a-zA-Z@]*\b|\\else\b|\\fi\b", block):
        raise KeyError(f"float {label} contains conditional TeX")
    return block


def _escaped(text, index):
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def float_caption(tex, label):
    """Source text of the unique caption in one labelled table."""
    block = float_block(tex, label)
    starts = [match.start() for match in re.finditer(r"\\caption\{", block)]
    if len(starts) != 1:
        raise KeyError(f"float {label} has {len(starts)} captions")
    content_start = starts[0] + len("\\caption{")
    depth = 1
    index = content_start
    while index < len(block) and depth:
        if block[index] == "{" and not _escaped(block, index):
            depth += 1
        elif block[index] == "}" and not _escaped(block, index):
            depth -= 1
        index += 1
    if depth:
        raise KeyError(f"unterminated caption in float {label}")
    return " ".join(block[content_start:index - 1].split())


def tabular_rows(tex, label):
    """Rows of the tabular belonging to the float carrying ``label``.

    Returns a mapping from each row's first cell to the list of remaining
    cells, both cleaned. Rows whose first cell is empty are keyed by the
    empty string, which is how a header row like ``& A & B`` arrives.
    """
    clean = strip_comments(tex)
    marker = "\\label{" + label + "}"
    occurrences = clean.count(marker)
    if occurrences == 0:
        raise KeyError(f"no float labelled {label}")
    if occurrences > 1:
        raise KeyError(f"{occurrences} floats labelled {label}; cannot bind")
    has_table = re.search(r"\\begin\{table\*?\}", clean)
    block = float_block(clean, label) if has_table else clean
    block = " ".join(block.split())
    if "\\begin{tabular}" not in block or "\\end{tabular}" not in block:
        raise KeyError(f"float {label} has no tabular")
    if block.count("\\begin{tabular}") != 1 or block.count("\\end{tabular}") != 1:
        raise KeyError(f"float {label} does not contain exactly one tabular")
    block = block[block.index("\\begin{tabular}"):block.index("\\end{tabular}")]
    # skip the column specification that follows \begin{tabular}
    body = block[block.index("}", block.index("\\begin{tabular}") + 15) + 1:]
    rows = {}
    duplicates = []
    for line in body.split("\\\\"):
        for rule in ("\\toprule", "\\midrule", "\\bottomrule"):
            line = line.replace(rule, "")
        # \cmidrule takes an optional trim and a column span, neither of
        # which is content; left in place it becomes a phantom row label.
        line = re.sub(r"\\cmidrule(\([lr]+\))?\{[^}]*\}", "", line)
        if not line.strip():
            continue
        cells = [clean_cell(cell) for cell in line.split("&")]
        if cells[0] in rows:
            # Header rows legitimately share an empty first cell; a repeated
            # real label is the dangerous case, since the second silently
            # replaces the first.
            if cells[0]:
                duplicates.append(cells[0])
            continue
        rows[cells[0]] = cells[1:]
    if duplicates:
        # A second row with the same label silently overwrites the first, so
        # a row TeX skips (inside \iffalse, say) could replace the visible
        # one. Refuse rather than pick.
        raise KeyError(f"duplicate row labels in {label}: {sorted(set(duplicates))}")
    return rows


def _normalise(text):
    for a, b in (("\u00b1", "+-"), ("\u2212", "-"), ("\u2013", "-"),
                 ("\ufb00", "ff"), ("\ufb01", "fi"), ("\ufb02", "fl"),
                 ("--", "-"), ("\u2009", ""), ("\u00a0", ""),
                 (",", ""), ("_", ""), (" ", "")):
        text = text.replace(a, b)
    return text.lower()


def assert_rendered(cells, pdf_text, context=""):
    """Return nonempty cell values absent from the normalized PDF text.

    A matching substring does not establish a value's row or table location.
    Callers that need those guarantees must delimit the table and check rows.
    An empty result means that all nonempty values have a substring match.
    """
    haystack = _normalise(pdf_text)
    missing = []
    for value in cells:
        if not value:
            continue
        if _normalise(value) not in haystack:
            missing.append(value)
    return missing
