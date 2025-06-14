"""Utilities for assertion debugging"""
import collections.abc
import pprint
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple

import _pytest._code
from _pytest import outcomes
from _pytest._io.saferepr import safeformat
from _pytest._io.saferepr import saferepr
from _pytest.compat import ATTRS_EQ_FIELD

# The _reprcompare attribute on the util module is used by the new assertion
# interpretation code and assertion rewriter to detect this plugin was
# loaded and in turn call the hooks defined here as part of the
# DebugInterpreter.
_reprcompare = None  # type: Optional[Callable[[str, object, object], Optional[str]]]

# Works similarly as _reprcompare attribute. Is populated with the hook call
# when pytest_runtest_setup is called.
_assertion_pass = None  # type: Optional[Callable[[int, str, str], None]]


def format_explanation(explanation: str) -> str:
    """This formats an explanation

    Normally all embedded newlines are escaped, however there are
    three exceptions: \n{, \n} and \n~.  The first two are intended
    cover nested explanations, see function and attribute explanations
    for examples (.visit_Call(), visit_Attribute()).  The last one is
    for when one explanation needs to span multiple lines, e.g. when
    displaying diffs.
    """
    lines = _split_explanation(explanation)
    result = _format_lines(lines)
    return "\n".join(result)


def _split_explanation(explanation: str) -> List[str]:
    """Return a list of individual lines in the explanation

    This will return a list of lines split on '\n{', '\n}' and '\n~'.
    Any other newlines will be escaped and appear in the line as the
    literal '\n' characters.
    """
    raw_lines = (explanation or "").split("\n")
    lines = [raw_lines[0]]
    for values in raw_lines[1:]:
        if values and values[0] in ["{", "}", "~", ">"]:
            lines.append(values)
        else:
            lines[-1] += "\\n" + values
    return lines


def _format_lines(lines: Sequence[str]) -> List[str]:
    """Format the individual lines

    This will replace the '{', '}' and '~' characters of our mini
    formatting language with the proper 'where ...', 'and ...' and ' +
    ...' text, taking care of indentation along the way.

    Return a list of formatted lines.
    """
    result = list(lines[:1])
    stack = [0]
    stackcnt = [0]
    for line in lines[1:]:
        if line.startswith("{"):
            if stackcnt[-1]:
                s = "and   "
            else:
                s = "where "
            stack.append(len(result))
            stackcnt[-1] += 1
            stackcnt.append(0)
            result.append(" +" + "  " * (len(stack) - 1) + s + line[1:])
        elif line.startswith("}"):
            stack.pop()
            stackcnt.pop()
            result[stack[-1]] += line[1:]
        else:
            assert line[0] in ["~", ">"]
            stack[-1] += 1
            indent = len(stack) if line.startswith("~") else len(stack) - 1
            result.append("  " * indent + line[1:])
    assert len(stack) == 1
    return result


def issequence(x: Any) -> bool:
    return isinstance(x, collections.abc.Sequence) and not isinstance(x, str)


def istext(x: Any) -> bool:
    return isinstance(x, str)


def isdict(x: Any) -> bool:
    return isinstance(x, dict)


def isset(x: Any) -> bool:
    return isinstance(x, (set, frozenset))


def isdatacls(obj: Any) -> bool:
    return getattr(obj, "__dataclass_fields__", None) is not None


def isattrs(obj: Any) -> bool:
    return getattr(obj, "__attrs_attrs__", None) is not None


def isiterable(obj: Any) -> bool:
    try:
        iter(obj)
        return not istext(obj)
    except TypeError:
        return False


def assertrepr_compare(config, op: str, left: Any, right: Any) -> Optional[List[str]]:
    """Return specialised explanations for some operators/operands"""
    verbose = config.getoption("verbose")
    if verbose > 1:
        left_repr = safeformat(left)
        right_repr = safeformat(right)
    else:
        # XXX: "15 chars indentation" is wrong
        #      ("E       AssertionError: assert "); should use term width.
        maxsize = (
            80 - 15 - len(op) - 2
        ) // 2  # 15 chars indentation, 1 space around op
        left_repr = saferepr(left, maxsize=maxsize)
        right_repr = saferepr(right, maxsize=maxsize)

    summary = "{} {} {}".format(left_repr, op, right_repr)

    explanation = None
    try:
        if op == "==":
            if istext(left) and istext(right):
                explanation = _diff_text(left, right, verbose)
            else:
                if issequence(left) and issequence(right):
                    explanation = _compare_eq_sequence(left, right, verbose)
                elif isset(left) and isset(right):
                    explanation = _compare_eq_set(left, right, verbose)
                elif isdict(left) and isdict(right):
                    explanation = _compare_eq_dict(left, right, verbose)
                elif type(left) == type(right) and (isdatacls(left) or isattrs(left)):
                    type_fn = (isdatacls, isattrs)
                    explanation = _compare_eq_cls(left, right, verbose, type_fn)
                elif verbose > 0:
                    explanation = _compare_eq_verbose(left, right)
                if isiterable(left) and isiterable(right):
                    expl = _compare_eq_iterable(left, right, verbose)
                    if explanation is not None:
                        explanation.extend(expl)
                    else:
                        explanation = expl
        elif op == "not in":
            if istext(left) and istext(right):
                explanation = _notin_text(left, right, verbose)
    except outcomes.Exit:
        raise
    except Exception:
        explanation = [
            "(pytest_assertion plugin: representation of details failed.  "
            "Probably an object has a faulty __repr__.)",
            str(_pytest._code.ExceptionInfo.from_current()),
        ]

    if not explanation:
        return None

    return [summary] + explanation


def _diff_text(left: str, right: str, verbose: int = 0) -> List[str]:
    """Return the explanation for the diff between text.

    Unless --verbose is used this will skip leading and trailing
    characters which are identical to keep the diff minimal.
    """
    from difflib import ndiff

    explanation = []  # type: List[str]

    if verbose < 1:
        i = 0  # just in case left or right has zero length
        for i in range(min(len(left), len(right))):
            if left[i] != right[i]:
                break
        if i > 42:
            i -= 10  # Provide some context
            explanation = [
                "Skipping %s identical leading characters in diff, use -v to show" % i
            ]
            left = left[i:]
            right = right[i:]
        if len(left) == len(right):
            for i in range(len(left)):
                if left[-i] != right[-i]:
                    break
            if i > 42:
                i -= 10  # Provide some context
                explanation += [
                    "Skipping {} identical trailing "
                    "characters in diff, use -v to show".format(i)
                ]
                left = left[:-i]
                right = right[:-i]
    keepends = True
    if left.isspace() or right.isspace():
        left = repr(str(left))
        right = repr(str(right))
        explanation += ["Strings contain only whitespace, escaping them using repr()"]
    explanation += [
        line.strip("\n")
        for line in ndiff(left.splitlines(keepends), right.splitlines(keepends))
    ]
    return explanation


def _compare_eq_verbose(left: Any, right: Any) -> List[str]:
    keepends = True
    left_lines = repr(left).splitlines(keepends)
    right_lines = repr(right).splitlines(keepends)

    explanation = []  # type: List[str]
    explanation += ["-" + line for line in left_lines]
    explanation += ["+" + line for line in right_lines]

    return explanation


def _surrounding_parens_on_own_lines(lines: List[str]) -> None:
    """Move opening/closing parenthesis/bracket to own lines."""
    opening = lines[0][:1]
    if opening in ["(", "[", "{"]:
        lines[0] = " " + lines[0][1:]
        lines[:] = [opening] + lines
    closing = lines[-1][-1:]
    if closing in [")", "]", "}"]:
        lines[-1] = lines[-1][:-1] + ","
        lines[:] = lines + [closing]


def _compare_eq_iterable(
    left: Iterable[Any], right: Iterable[Any], verbose: int = 0
) -> List[str]:
    if not verbose:
        return ["Use -v to get the full diff"]
    # dynamic import to speedup pytest
    import difflib

    left_formatting = pprint.pformat(left).splitlines()
    right_formatting = pprint.pformat(right).splitlines()

    # Re-format for different output lengths.
    lines_left = len(left_formatting)
    lines_right = len(right_formatting)
    if lines_left != lines_right:
        if lines_left > lines_right:
            max_width = min(len(x) for x in left_formatting)
        else:
            max_width = min(len(x) for x in right_formatting)

        right_formatting = pprint.pformat(right, width=max_width).splitlines()
        lines_right = len(right_formatting)
        left_formatting = pprint.pformat(left, width=max_width).splitlines()
        lines_left = len(left_formatting)

    if lines_left > 1 or lines_right > 1:
        _surrounding_parens_on_own_lines(left_formatting)
        _surrounding_parens_on_own_lines(right_formatting)

    explanation = ["Full diff:"]
    explanation.extend(
        line.rstrip() for line in difflib.ndiff(left_formatting, right_formatting)
    )
    return explanation


def _compare_eq_sequence(
    left: Sequence[Any], right: Sequence[Any], verbose: int = 0
) -> List[str]:
    comparing_bytes = isinstance(left, bytes) and isinstance(right, bytes)
    explanation = []  # type: List[str]
    len_left = len(left)
    len_right = len(right)
    for i in range(min(len_left, len_right)):
        if left[i] != right[i]:
            if comparing_bytes:
                # when comparing bytes, we want to see their ascii representation
                # instead of their numeric values (#5260)
                # using a slice gives us the ascii representation:
                # >>> s = b'foo'
                # >>> s[0]
                # 102
                # >>> s[0:1]
                # b'f'
                left_value = left[i : i + 1]
                right_value = right[i : i + 1]
            else:
                left_value = left[i]
                right_value = right[i]

            explanation += [
                "At index {} diff: {!r} != {!r}".format(i, left_value, right_value)
            ]
            break

    if comparing_bytes:
        # when comparing bytes, it doesn't help to show the "sides contain one or more items"
        # longer explanation, so skip it
        return explanation

    len_diff = len_left - len_right
    if len_diff:
        if len_diff > 0:
            dir_with_more = "Left"
            extra = saferepr(left[len_right])
        else:
            len_diff = 0 - len_diff
            dir_with_more = "Right"
            extra = saferepr(right[len_left])

        if len_diff == 1:
            explanation += [
                "{} contains one more item: {}".format(dir_with_more, extra)
            ]
        else:
            explanation += [
                "%s contains %d more items, first extra item: %s"
                % (dir_with_more, len_diff, extra)
            ]
    return explanation


def _compare_eq_set(
    left: AbstractSet[Any], right: AbstractSet[Any], verbose: int = 0
) -> List[str]:
    explanation = []
    diff_left = left - right
    diff_right = right - left
    if diff_left:
        explanation.append("Extra items in the left set:")
        for item in diff_left:
            explanation.append(saferepr(item))
    if diff_right:
        explanation.append("Extra items in the right set:")
        for item in diff_right:
            explanation.append(saferepr(item))
    return explanation


def _compare_eq_dict(
    left: Mapping[Any, Any], right: Mapping[Any, Any], verbose: int = 0
) -> List[str]:
    explanation = []  # type: List[str]
    set_left = set(left)
    set_right = set(right)
    common = set_left.intersection(set_right)
    same = {k: left[k] for k in common if left[k] == right[k]}
    if same and verbose < 2:
        explanation += ["Omitting %s identical items, use -vv to show" % len(same)]
    elif same:
        explanation += ["Common items:"]
        explanation += pprint.pformat(same).splitlines()
    diff = {k for k in common if left[k] != right[k]}
    if diff:
        explanation += ["Differing items:"]
        for k in diff:
            explanation += [saferepr({k: left[k]}) + " != " + saferepr({k: right[k]})]
    extra_left = set_left - set_right
    len_extra_left = len(extra_left)
    if len_extra_left:
        explanation.append(
            "Left contains %d more item%s:"
            % (len_extra_left, "" if len_extra_left == 1 else "s")
        )
        explanation.extend(
            pprint.pformat({k: left[k] for k in extra_left}).splitlines()
        )
    extra_right = set_right - set_left
    len_extra_right = len(extra_right)
    if len_extra_right:
        explanation.append(
            "Right contains %d more item%s:"
            % (len_extra_right, "" if len_extra_right == 1 else "s")
        )
        explanation.extend(
            pprint.pformat({k: right[k] for k in extra_right}).splitlines()
        )
    return explanation


def _compare_eq_cls(
    left: Any,
    right: Any,
    verbose: int,
    type_fns: Tuple[Callable[[Any], bool], Callable[[Any], bool]],
) -> List[str]:
    isdatacls, isattrs = type_fns
    if isdatacls(left):
        all_fields = left.__dataclass_fields__
        fields_to_check = [field for field, info in all_fields.items() if info.compare]
    elif isattrs(left):
        all_fields = left.__attrs_attrs__
        fields_to_check = [
            field.name for field in all_fields if getattr(field, ATTRS_EQ_FIELD)
        ]

    same = []
    diff = []
    for field in fields_to_check:
        if getattr(left, field) == getattr(right, field):
            same.append(field)
        else:
            diff.append(field)

    explanation = []
    if same and verbose < 2:
        explanation.append("Omitting %s identical items, use -vv to show" % len(same))
    elif same:
        explanation += ["Matching attributes:"]
        explanation += pprint.pformat(same).splitlines()
    if diff:
        explanation += ["Differing attributes:"]
        for field in diff:
            explanation += [
                ("%s: %r != %r") % (field, getattr(left, field), getattr(right, field))
            ]
    return explanation


def _notin_text(term: str, text: str, verbose: int = 0) -> List[str]:
    index = text.find(term)
    head = text[:index]
    tail = text[index + len(term) :]
    correct_text = head + tail
    diff = _diff_text(correct_text, text, verbose)
    newdiff = ["%s is contained here:" % saferepr(term, maxsize=42)]
    for line in diff:
        if line.startswith("Skipping"):
            continue
        if line.startswith("- "):
            continue
        if line.startswith("+ "):
            newdiff.append("  " + line[2:])
        else:
            newdiff.append(line)
    return newdiff
