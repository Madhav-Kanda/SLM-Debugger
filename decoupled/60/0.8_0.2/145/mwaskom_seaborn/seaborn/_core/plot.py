"""The classes for specifying and compiling a declarative visualization."""
from __future__ import annotations

import io
import os
import re
import sys
import inspect
import itertools
import textwrap
from contextlib import contextmanager
from collections import abc
from collections.abc import Callable, Generator
from typing import Any, List, Optional, cast

from cycler import cycler
import pandas as pd
from pandas import DataFrame, Series, Index
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.figure import Figure

from seaborn._marks.base import Mark
from seaborn._stats.base import Stat
from seaborn._core.data import PlotData
from seaborn._core.moves import Move
from seaborn._core.scales import Scale, Nominal
from seaborn._core.subplots import Subplots
from seaborn._core.groupby import GroupBy
from seaborn._core.properties import PROPERTIES, Property
from seaborn._core.typing import (
    DataSource,
    VariableSpec,
    VariableSpecList,
    OrderSpec,
    Default,
)
from seaborn._core.exceptions import PlotSpecError
from seaborn._core.rules import categorical_order
from seaborn._compat import set_scale_obj, set_layout_engine
from seaborn.rcmod import axes_style, plotting_context
from seaborn.palettes import color_palette
from seaborn.external.version import Version

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from matplotlib.figure import SubFigure


if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


default = Default()


# ---- Definitions for internal specs --------------------------------- #


class Layer(TypedDict, total=False):

    mark: Mark  # TODO allow list?
    stat: Stat | None  # TODO allow list?
    move: Move | list[Move] | None
    data: PlotData
    source: DataSource
    vars: dict[str, VariableSpec]
    orient: str
    legend: bool


class FacetSpec(TypedDict, total=False):

    variables: dict[str, VariableSpec]
    structure: dict[str, list[str]]
    wrap: int | None


class PairSpec(TypedDict, total=False):

    variables: dict[str, VariableSpec]
    structure: dict[str, list[str]]
    cross: bool
    wrap: int | None


# --- Local helpers ----------------------------------------------------------------


@contextmanager
def theme_context(params: dict[str, Any]) -> Generator:
    """Temporarily modify specifc matplotlib rcParams."""
    orig_params = {k: mpl.rcParams[k] for k in params}
    color_codes = "bgrmyck"
    nice_colors = [*color_palette("deep6"), (.15, .15, .15)]
    orig_colors = [mpl.colors.colorConverter.colors[x] for x in color_codes]
    # TODO how to allow this to reflect the color cycle when relevant?
    try:
        mpl.rcParams.update(params)
        for (code, color) in zip(color_codes, nice_colors):
            mpl.colors.colorConverter.colors[code] = color
            mpl.colors.colorConverter.cache[code] = color
        yield
    finally:
        mpl.rcParams.update(orig_params)
        for (code, color) in zip(color_codes, orig_colors):
            mpl.colors.colorConverter.colors[code] = color
            mpl.colors.colorConverter.cache[code] = color


def build_plot_signature(cls):
    """
    Decorator function for giving Plot a useful signature.

    Currently this mostly saves us some duplicated typing, but we would
    like eventually to have a way of registering new semantic properties,
    at which point dynamic signature generation would become more important.

    """
    sig = inspect.signature(cls)
    params = [
        inspect.Parameter("args", inspect.Parameter.VAR_POSITIONAL),
        inspect.Parameter("data", inspect.Parameter.KEYWORD_ONLY, default=None)
    ]
    params.extend([
        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None)
        for name in PROPERTIES
    ])
    new_sig = sig.replace(parameters=params)
    cls.__signature__ = new_sig

    known_properties = textwrap.fill(
        ", ".join([f"|{p}|" for p in PROPERTIES]),
        width=78, subsequent_indent=" " * 8,
    )

    if cls.__doc__ is not None:  # support python -OO mode
        cls.__doc__ = cls.__doc__.format(known_properties=known_properties)

    return cls


# ---- The main interface for declarative plotting -------------------- #


@build_plot_signature
class Plot:
    """
    An interface for declaratively specifying statistical graphics.

    Plots are constructed by initializing this class and adding one or more
    layers, comprising a `Mark` and optional `Stat` or `Move`.  Additionally,
    faceting variables or variable pairings may be defined to divide the space
    into multiple subplots. The mappings from data values to visual properties
    can be parametrized using scales, although the plot will try to infer good
    defaults when scales are not explicitly defined.

    The constructor accepts a data source (a :class:`pandas.DataFrame` or
    dictionary with columnar values) and variable assignments. Variables can be
    passed as keys to the data source or directly as data vectors.  If multiple
    data-containing objects are provided, they will be index-aligned.

    The data source and variables defined in the constructor will be used for
    all layers in the plot, unless overridden or disabled when adding a layer.

    The following variables can be defined in the constructor:
        {known_properties}

    The `data`, `x`, and `y` variables can be passed as positional arguments or
    using keywords. Whether the first positional argument is interpreted as a
    data source or `x` variable depends on its type.

    The methods of this class return a copy of the instance; use chaining to
    build up a plot through multiple calls. Methods can be called in any order.

    Most methods only add information to the plot spec; no actual processing
    happens until the plot is shown or saved. It is also possible to compile
    the plot without rendering it to access the lower-level representation.

    """
    _data: PlotData
    _layers: list[Layer]

    _scales: dict[str, Scale]
    _shares: dict[str, bool | str]
    _limits: dict[str, tuple[Any, Any]]
    _labels: dict[str, str | Callable[[str], str]]
    _theme: dict[str, Any]

    _facet_spec: FacetSpec
    _pair_spec: PairSpec

    _figure_spec: dict[str, Any]
    _subplot_spec: dict[str, Any]
    _layout_spec: dict[str, Any]

    def __init__(
        self,
        *args: DataSource | VariableSpec,
        data: DataSource = None,
        **variables: VariableSpec,
    ):

        if args:
            data, variables = self._resolve_positionals(args, data, variables)

        unknown = [x for x in variables if x not in PROPERTIES]
        if unknown:
            err = f"Plot() got unexpected keyword argument(s): {', '.join(unknown)}"
            raise TypeError(err)

        self._data = PlotData(data, variables)

        self._layers = []

        self._scales = {}
        self._shares = {}
        self._limits = {}
        self._labels = {}
        self._theme = {}

        self._facet_spec = {}
        self._pair_spec = {}

        self._figure_spec = {}
        self._subplot_spec = {}
        self._layout_spec = {}

        self._target = None

    def _resolve_positionals(
        self,
        args: tuple[DataSource | VariableSpec, ...],
        data: DataSource,
        variables: dict[str, VariableSpec],
    ) -> tuple[DataSource, dict[str, VariableSpec]]:
        """Handle positional arguments, which may contain data / x / y."""
        if len(args) > 3:
            err = "Plot() accepts no more than 3 positional arguments (data, x, y)."
            raise TypeError(err)

        # TODO need some clearer way to differentiate data / vector here
        # (There might be an abstract DataFrame class to use here?)
        if isinstance(args[0], (abc.Mapping, pd.DataFrame)):
            if data is not None:
                raise TypeError("`data` given by both name and position.")
            data, args = args[0], args[1:]

        if len(args) == 2:
            x, y = args
        elif len(args) == 1:
            x, y = *args, None
        else:
            x = y = None

        for name, var in zip("yx", (y, x)):
            if var is not None:
                if name in variables:
                    raise TypeError(f"`{name}` given by both name and position.")
                # Keep coordinates at the front of the variables dict
                # Cast type because we know this isn't a DataSource at this point
                variables = {name: cast(VariableSpec, var), **variables}

        return data, variables

    def __add__(self, other):

        if isinstance(other, Mark) or isinstance(other, Stat):
            raise TypeError("Sorry, this isn't ggplot! Perhaps try Plot.add?")

        other_type = other.__class__.__name__
        raise TypeError(f"Unsupported operand type(s) for +: 'Plot' and '{other_type}")

    def _repr_png_(self) -> tuple[bytes, dict[str, float]]:

        return self.plot()._repr_png_()

    # TODO _repr_svg_?

    def _clone(self) -> Plot:
        """Generate a new object with the same information as the current spec."""
        new = Plot()

        # TODO any way to enforce that data does not get mutated?
        new._data = self._data

        new._layers.extend(self._layers)

        new._scales.update(self._scales)
        new._shares.update(self._shares)
        new._limits.update(self._limits)
        new._labels.update(self._labels)
        new._theme.update(self._theme)

        new._facet_spec.update(self._facet_spec)
        new._pair_spec.update(self._pair_spec)

        new._figure_spec.update(self._figure_spec)
        new._subplot_spec.update(self._subplot_spec)
        new._layout_spec.update(self._layout_spec)

        new._target = self._target

        return new

    def _theme_with_defaults(self) -> dict[str, Any]:

        style_groups = [
            "axes", "figure", "font", "grid", "hatch", "legend", "lines",
            "mathtext", "markers", "patch", "savefig", "scatter",
            "xaxis", "xtick", "yaxis", "ytick",
        ]
        base = {
            k: mpl.rcParamsDefault[k] for k in mpl.rcParams
            if any(k.startswith(p) for p in style_groups)
        }
        theme = {
            **base,
            **axes_style("darkgrid"),
            **plotting_context("notebook"),
            "axes.prop_cycle": cycler("color", color_palette("deep")),
        }
        theme.update(self._theme)
        return theme

    @property
    def _variables(self) -> list[str]:

        variables = (
            list(self._data.frame)
            + list(self._pair_spec.get("variables", []))
            + list(self._facet_spec.get("variables", []))
        )
        for layer in self._layers:
            variables.extend(v for v in layer["vars"] if v not in variables)

        # Coerce to str in return to appease mypy; we know these will only
        # ever be strings but I don't think we can type a DataFrame that way yet
        return [str(v) for v in variables]

    def on(self, target: Axes | SubFigure | Figure) -> Plot:
        """
        Provide existing Matplotlib figure or axes for drawing the plot.

        When using this method, you will also need to explicitly call a method that
        triggers compilation, such as :meth:`Plot.show` or :meth:`Plot.save`. If you
        want to postprocess using matplotlib, you'd need to call :meth:`Plot.plot`
        first to compile the plot without rendering it.

        Parameters
        ----------
        target : Axes, SubFigure, or Figure
            Matplotlib object to use. Passing :class:`matplotlib.axes.Axes` will add
            artists without otherwise modifying the figure. Otherwise, subplots will be
            created within the space of the given :class:`matplotlib.figure.Figure` or
            :class:`matplotlib.figure.SubFigure`.

        Examples
        --------
        .. include:: ../docstrings/objects.Plot.on.rst

        """
        accepted_types: tuple  # Allow tuple of various length
        if hasattr(mpl.figure, "SubFigure"):  # Added in mpl 3.4
            accepted_types = (
                mpl.axes.Axes, mpl.figure.SubFigure, mpl.figure.Figure
            )
            accepted_types_str = (
                f"{mpl.axes.Axes}, {mpl.figure.SubFigure}, or {mpl.figure.Figure}"
            )
        else:
            accepted_types = mpl.axes.Axes, mpl.figure.Figure
            accepted_types_str = f"{mpl.axes.Axes} or {mpl.figure.Figure}"

        if not isinstance(target, accepted_types):
            err = (
                f"The `Plot.on` target must be an instance of {accepted_types_str}. "
                f"You passed an instance of {target.__class__} instead."
            )
            raise TypeError(err)

        new = self._clone()
        new._target = target

        return new

    def add(
        self,
        mark: Mark,
        *transforms: Stat | Mark,
        orient: str | None = None,
        legend: bool = True,
        data: DataSource = None,
        **variables: VariableSpec,
    ) -> Plot:
        """
        Specify a layer of the visualization in terms of mark and data transform(s).

        This is the main method for specifying how the data should be visualized.
        It can be called multiple times with different arguments to define
        a plot with multiple layers.

        Parameters
        ----------
        mark : :class:`Mark`
            The visual representation of the data to use in this layer.
        transforms : :class:`Stat` or :class:`Move`
            Objects representing transforms to be applied before plotting the data.
            Currently, at most one :class:`Stat` can be used, and it
            must be passed first. This constraint will be relaxed in the future.
        orient : "x", "y", "v", or "h"
            The orientation of the mark, which also affects how transforms are computed.
            Typically corresponds to the axis that defines groups for aggregation.
            The "v" (vertical) and "h" (horizontal) options are synonyms for "x" / "y",
            but may be more intuitive with some marks. When not provided, an
            orientation will be inferred from characteristics of the data and scales.
        legend : bool
            Option to suppress the mark/mappings for this layer from the legend.
        data : DataFrame or dict
            Data source to override the global source provided in the constructor.
        variables : data vectors or identifiers
            Additional layer-specific variables, including variables that will be
            passed directly to the transforms without scaling.

        Examples
        --------
        .. include:: ../docstrings/objects.Plot.add.rst

        """
        if not isinstance(mark, Mark):
            msg = f"mark must be a Mark instance, not {type(mark)!r}."
            raise TypeError(msg)

        # TODO This API for transforms was a late decision, and previously Plot.add
        # accepted 0 or 1 Stat instances and 0, 1, or a list of Move instances.
        # It will take some work to refactor the internals so that Stat and Move are
        # treated identically, and until then well need to "unpack" the transforms
        # here and enforce limitations on the order / types.

        stat: Optional[Stat]
        move: Optional[List[Move]]
        error = False
        if not transforms:
            stat, move = None, None
        elif isinstance(transforms[0], Stat):
            stat = transforms[0]
            move = [m for m in transforms[1:] if isinstance(m, Move)]
            error = len(move) != len(transforms) - 1
        else:
            stat = None
            move = [m for m in transforms if isinstance(m, Move)]
            error = len(move) != len(transforms)

        if error:
            msg = " ".join([
                "Transforms must have at most one Stat type (in the first position),",
                "and all others must be a Move type. Given transform type(s):",
                ", ".join(str(type(t).__name__) for t in transforms) + "."
            ])
            raise TypeError(msg)

        new = self._clone()
        new._layers.append({
            "mark": mark,
            "stat": stat,
            "move": move,
            # TODO it doesn't work to supply scalars to variables, but it should
            "vars": variables,
            "source": data,
            "legend": legend,
            "orient": {"v": "x", "h": "y"}.get(orient, orient),  # type: ignore
        })

        return new

    def pair(
        self,
        x: VariableSpecList = None,
        y: VariableSpecList = None,
        wrap: int | None = None,
        cross: bool = True,
    ) -> Plot:
        """
        Produce subplots by pairing multiple `x` and/or `y` variables.

        Parameters
        ----------
        x, y : sequence(s) of data vectors or identifiers
            Variables that will define the grid of subplots.
        wrap : int
            When using only `x` or `y`, "wrap" subplots across a two-dimensional grid
            with this many columns (when using `x`) or rows (when using `y`).
        cross : bool
            When False, zip the `x` and `y` lists such that the first subplot gets the
            first pair, the second gets the second pair, etc. Otherwise, create a
            two-dimensional grid from the cartesian product of the lists.

        Examples
        --------
        .. include:: ../docstrings/objects.Plot.pair.rst

        """
        # TODO Add transpose= arg, which would then draw pair(y=[...]) across rows
        # This may also be possible by setting `wrap=1`, but is that too unobvious?
        # TODO PairGrid features not currently implemented: diagonals, corner

        pair_spec: PairSpec = {}

        axes = {"x": [] if x is None else x, "y": [] if y is None else y}
        for axis, arg in axes.items():
            if isinstance(arg, (str, int)):
                err = f"You must pass a sequence of variable keys to `{axis}`"
                raise TypeError(err)

        pair_spec["variables"] = {}
        pair_spec["structure"] = {}

        for axis in "xy":
            keys = []
            for i, col in enumerate(axes[axis]):
                key = f"{axis}{i}"
                keys.append(key)
                pair_spec["variables"][key] = col

            if keys:
                pair_spec["structure"][axis] = keys

        if not cross and len(axes["x"]) != len(axes["y"]):
            err = "Lengths of the `x` and `y` lists must match with cross=False"
            raise ValueError(err)

        pair_spec["cross"] = cross
        pair_spec["wrap"] = wrap

        new = self._clone()
        new._pair_spec.update(pair_spec)
        return new

    def facet(
        self,
        col: VariableSpec = None,
        row: VariableSpec = None,
        order: OrderSpec | dict[str, OrderSpec] = None,
        wrap: int | None = None,
    ) -> Plot:
        """
        Produce subplots with conditional subsets of the data.

        Parameters
        ----------
        col, row : data vectors or identifiers
            Variables used to define subsets along the columns and/or rows of the grid.
            Can be references to the global data source passed in the constructor.
        order : list of strings, or dict with dimensional keys
            Define the order of the faceting variables.
        wrap : int
            When using only `col` or `row`, wrap subplots across a two-dimensional
            grid with this many subplots on the faceting dimension.

        Examples
        --------
        .. include:: ../docstrings/objects.Plot.facet.rst

        """
        variables: dict[str, VariableSpec] = {}
        if col is not None:
            variables["col"] = col
        if row is not None:
            variables["row"] = row

        structure = {}
        if isinstance(order, dict):
            for dim in ["col", "row"]:
                dim_order = order.get(dim)
                if dim_order is not None:
                    structure[dim] = list(dim_order)
        elif order is not None:
            if col is not None and row is not None:
                err = " ".join([
                    "When faceting on both col= and row=, passing `order` as a list"
                    "is ambiguous. Use a dict with 'col' and/or 'row' keys instead."
                ])
                raise RuntimeError(err)
            elif col is not None:
                structure["col"] = list(order)
            elif row is not None:
                structure["row"] = list(order)

        spec: FacetSpec = {
            "variables": variables,
            "structure": structure,
            "wrap": wrap,
        }

        new = self._clone()
        new._facet_spec.update(spec)

        return new

    # TODO def twin()?

    def scale(self, **scales: Scale) -> Plot:
        """
        Specify mappings from data units to visual properties.

        Keywords correspond to variables defined in the plot, including coordinate
        variables (`x`, `y`) and semantic variables (`color`, `pointsize`, etc.).

        A number of "magic" arguments are accepted, including:
            - The name of a transform (e.g., `"log"`, `"sqrt"`)
            - The name of a palette (e.g., `"viridis"`, `"muted"`)
            - A tuple of values, defining the output range (e.g. `(1, 5)`)
            - A dict, implying a :class:`Nominal` scale (e.g. `{"a": .2, "b": .5}`)
            - A list of values, implying a :class:`Nominal` scale (e.g. `["b", "r"]`)

        For more explicit control, pass a scale spec object such as :class:`Continuous`
        or :class:`Nominal`. Or pass `None` to use an "identity" scale, which treats
        data values as literally encoding visual properties.

        Examples
        --------
        .. include:: ../docstrings/objects.Plot.scale.rst

        """
        new = self._clone()
        new._scales.update(scales)
        return new

    def share(self, **shares: bool | str) -> Plot:
        """
        Control sharing of axis limits and ticks across subplots.

        Keywords correspond to variables defined in the plot, and values can be
        boolean (to share across all subplots), or one of "row" or "col" (to share
        more selectively across one dimension of a grid).

        Behavior for non-coordinate variables is currently undefined.

        Examples
        --------
        .. include:: ../docstrings/objects.Plot.share.rst

        """
        new = self._clone()
        new._shares.update(shares)
        return new

    def limit(self, **limits: tuple[Any, Any]) -> Plot:
        """
        Control the range of visible data.

        Keywords correspond to variables defined in the plot, and values are a
        `(min, max)` tuple (where either can be `None` to leave unset).

        Limits apply only to the axis; data outside the visible range are
        still used for any stat transforms and added to the plot.

        Behavior for non-coordinate variables is currently undefined.

        Examples
        --------
        .. include:: ../docstrings/objects.Plot.limit.rst

        """
        new = self._clone()
        new._limits.update(limits)
        return new

    def label(self, *, title=None, **variables: str | Callable[[str], str]) -> Plot:
        """
        Control the labels and titles for axes, legends, and subplots.

        Additional keywords correspond to variables defined in the plot.
        Values can be one of the following types:

        - string (used literally; pass "" to clear the default label)
        - function (called on the default label)

        For coordinate variables, the value sets the axis label.
        For semantic variables, the value sets the legend title.
        For faceting variables, `title=` modifies the subplot-specific label,
        while `col=` and/or `row=` add a label for the faceting variable.
        When using a single subplot, `title=` sets its title.

        Examples
        --------
        .. include:: ../docstrings/objects.Plot.label.rst


        """
        new = self._clone()
        if title is not None:
            new._labels["title"] = title
        new._labels.update(variables)
        return new

    def layout(
        self,
        *,
        size: tuple[float, float] | Default = default,
        engine: str | None | Default = default,
    ) -> Plot:
        """
        Control the figure size and layout.

        .. note::

            Default figure sizes and the API for specifying the figure size are subject
            to change in future "experimental" releases of the objects API. The default
            layout engine may also change.

        Parameters
        ----------
        size : (width, height)
            Size of the resulting figure, in inches. Size is inclusive of legend when
            using pyplot, but not otherwise.
        engine : {{"tight", "constrained", None}}
            Name of method for automatically adjusting the layout to remove overlap.
            The default depends on whether :meth:`Plot.on` is used.

        Examples
        --------
        .. include:: ../docstrings/objects.Plot.layout.rst

        """
        # TODO add an "auto" mode for figsize that roughly scales with the rcParams
        # figsize (so that works), but expands to prevent subplots from being squished
        # Also should we have height=, aspect=, exclusive with figsize? Or working
        # with figsize when only one is defined?

        new = self._clone()

        if size is not default:
            new._figure_spec["figsize"] = size
        if engine is not default:
            new._layout_spec["engine"] = engine

        return new

    # TODO def legend (ugh)

    def theme(self, *args: dict[str, Any]) -> Plot:
        """
        Control the default appearance of elements in the plot.

        .. note::

            The API for customizing plot appearance is not yet finalized.
            Currently, the only valid argument is a dict of matplotlib rc parameters.
            (This dict must be passed as a positional argument.)

            It is likely that this method will be enhanced in future releases.

        Matplotlib rc parameters are documented on the following page:
        https://matplotlib.org/stable/tutorials/introductory/customizing.html

        Examples
        --------
        .. include:: ../docstrings/objects.Plot.theme.rst

        """
        new = self._clone()

        # We can skip this whole block on Python 3.8+ with positional-only syntax
        nargs = len(args)
        if nargs != 1:
            err = f"theme() takes 1 positional argument, but {nargs} were given"
            raise TypeError(err)

        rc = args[0]
        new._theme.update(rc)

        return new

    def save(self, loc, **kwargs) -> Plot:
        """
        Compile the plot and write it to a buffer or file on disk.

        Parameters
        ----------
        loc : str, path, or buffer
            Location on disk to save the figure, or a buffer to write into.
        kwargs
            Other keyword arguments are passed through to
            :meth:`matplotlib.figure.Figure.savefig`.

        """
        # TODO expose important keyword arguments in our signature?
        with theme_context(self._theme_with_defaults()):
            self._plot().save(loc, **kwargs)
        return self

    def show(self, **kwargs) -> None:
        """
        Compile the plot and display it by hooking into pyplot.

        Calling this method is not necessary to render a plot in notebook context,
        but it may be in other environments (e.g., in a terminal). After compiling the
        plot, it calls :func:`matplotlib.pyplot.show` (passing any keyword parameters).

        Unlike other :class:`Plot` methods, there is no return value. This should be
        the last method you call when specifying a plot.

        """
        # TODO make pyplot configurable at the class level, and when not using,
        # import IPython.display and call on self to populate cell output?

        # Keep an eye on whether matplotlib implements "attaching" an existing
        # figure to pyplot: https://github.com/matplotlib/matplotlib/pull/14024

        self.plot(pyplot=True).show(**kwargs)

    def plot(self, pyplot: bool = False) -> Plotter:
        """
        Compile the plot spec and return the Plotter object.
        """
        with theme_context(self._theme_with_defaults()):
            return self._plot(pyplot)

    def _plot(self, pyplot: bool = False) -> Plotter:

        # TODO if we have _target object, pyplot should be determined by whether it
        # is hooked into the pyplot state machine (how do we check?)

        plotter = Plotter(pyplot=pyplot, theme=self._theme_with_defaults())

        # Process the variable assignments and initialize the figure
        common, layers = plotter._extract_data(self)
        plotter._setup_figure(self, common, layers)

        # Process the scale spec for coordinate variables and transform their data
        coord_vars = [v for v in self._variables if re.match(r"^x|y", v)]
        plotter._setup_scales(self, common, layers, coord_vars)

        # Apply statistical transform(s)
        plotter._compute_stats(self, layers)

        # Process scale spec for semantic variables and coordinates computed by stat
        plotter._setup_scales(self, common, layers)

        # TODO Remove these after updating other methods
        # ---- Maybe have debug= param that attaches these when True?
        plotter._data = common
        plotter._layers = layers

        # Process the data for each layer and add matplotlib artists
        for layer in layers:
            plotter._plot_layer(self, layer)

        # Add various figure decorations
        plotter._make_legend(self)
        plotter._finalize_figure(self)

        return plotter


# ---- The plot compilation engine ---------------------------------------------- #


class Plotter:
    """
    Engine for compiling a :class:`Plot` spec into a Matplotlib figure.

    This class is not intended to be instantiated directly by users.

    """
    # TODO decide if we ever want these (Plot.plot(debug=True))?
    _data: PlotData
    _layers: list[Layer]
    _figure: Figure

    def __init__(self, pyplot: bool, theme: dict[str, Any]):

        self._pyplot = pyplot
        self._theme = theme
        self._legend_contents: list[tuple[
            tuple[str, str | int], list[Artist], list[str],
        ]] = []
        self._scales: dict[str, Scale] = {}

    def save(self, loc, **kwargs) -> Plotter:  # TODO type args
        kwargs.setdefault("dpi", 96)
        try:
            loc = os.path.expanduser(loc)
        except TypeError:
            # loc may be a buffer in which case that would not work
            pass
        self._figure.savefig(loc, **kwargs)
        return self

    def show(self, **kwargs) -> None:
        """
        Display the plot by hooking into pyplot.

        This method calls :func:`matplotlib.pyplot.show` with any keyword parameters.

        """
        # TODO if we did not create the Plotter with pyplot, is it possible to do this?
        # If not we should clearly raise.
        import matplotlib.pyplot as plt
        with theme_context(self._theme):
            plt.show(**kwargs)

    # TODO API for accessing the underlying matplotlib objects
    # TODO what else is useful in the public API for this class?

    def _repr_png_(self) -> tuple[bytes, dict[str, float]]:

        # TODO better to do this through a Jupyter hook? e.g.
        # ipy = IPython.core.formatters.get_ipython()
        # fmt = ipy.display_formatter.formatters["text/html"]
        # fmt.for_type(Plot, ...)
        # Would like to have a svg option too, not sure how to make that flexible

        # TODO use matplotlib backend directly instead of going through savefig?

        # TODO perhaps have self.show() flip a switch to disable this, so that
        # user does not end up with two versions of the figure in the output

        # TODO use bbox_inches="tight" like the inline backend?
        # pro: better results,  con: (sometimes) confusing results
        # Better solution would be to default (with option to change)
        # to using constrained/tight layout.

        # TODO need to decide what the right default behavior here is:
        # - Use dpi=72 to match default InlineBackend figure size?
        # - Accept a generic "scaling" somewhere and scale DPI from that,
        #   either with 1x -> 72 or 1x -> 96 and the default scaling be .75?
        # - Listen to rcParams? InlineBackend behavior makes that so complicated :(
        # - Do we ever want to *not* use retina mode at this point?

        from PIL import Image

        dpi = 96
        buffer = io.BytesIO()

        with theme_context(self._theme):
            self._figure.savefig(buffer, dpi=dpi * 2, format="png", bbox_inches="tight")
        data = buffer.getvalue()

        scaling = .85 / 2
        w, h = Image.open(buffer).size
        metadata = {"width": w * scaling, "height": h * scaling}
        return data, metadata

    def _extract_data(self, p: Plot) -> tuple[PlotData, list[Layer]]:

        common_data = (
            p._data
            .join(None, p._facet_spec.get("variables"))
            .join(None, p._pair_spec.get("variables"))
        )

        layers: list[Layer] = []
        for layer in p._layers:
            spec = layer.copy()
            spec["data"] = common_data.join(layer.get("source"), layer.get("vars"))
            layers.append(spec)

        return common_data, layers

    def _resolve_label(self, p: Plot, var: str, auto_label: str | None) -> str:

        label: str
        if var in p._labels:
            manual_label = p._labels[var]
            if callable(manual_label) and auto_label is not None:
                label = manual_label(auto_label)
            else:
                label = cast(str, manual_label)
        elif auto_label is None:
            label = ""
        else:
            label = auto_label
        return label

    def _setup_figure(self, p: Plot, common: PlotData, layers: list[Layer]) -> None:

        # --- Parsing the faceting/pairing parameterization to specify figure grid

        subplot_spec = p._subplot_spec.copy()
        facet_spec = p._facet_spec.copy()
        pair_spec = p._pair_spec.copy()

        for axis in "xy":
            if axis in p._shares:
                subplot_spec[f"share{axis}"] = p._shares[axis]

        for dim in ["col", "row"]:
            if dim in common.frame and dim not in facet_spec["structure"]:
                order = categorical_order(common.frame[dim])
                facet_spec["structure"][dim] = order

        self._subplots = subplots = Subplots(subplot_spec, facet_spec, pair_spec)

        # --- Figure initialization
        self._figure = subplots.init_figure(
            pair_spec, self._pyplot, p._figure_spec, p._target,
        )

        # --- Figure annotation
        for sub in subplots:
            ax = sub["ax"]
            for axis in "xy":
                axis_key = sub[axis]

                # ~~ Axis labels

                # TODO Should we make it possible to use only one x/y label for
                # all rows/columns in a faceted plot? Maybe using sub{axis}label,
                # although the alignments of the labels from that method leaves
                # something to be desired (in terms of how it defines 'centered').
                names = [
                    common.names.get(axis_key),
                    *(layer["data"].names.get(axis_key) for layer in layers)
                ]
                auto_label = next((name for name in names if name is not None), None)
                label = self._resolve_label(p, axis_key, auto_label)
                ax.set(**{f"{axis}label": label})

                # ~~ Decoration visibility

                # TODO there should be some override (in Plot.layout?) so that
                # axis / tick labels can be shown on interior shared axes if desired

                axis_obj = getattr(ax, f"{axis}axis")
                visible_side = {"x": "bottom", "y": "left"}.get(axis)
                show_axis_label = (
                    sub[visible_side]
                    or not p._pair_spec.get("cross", True)
                    or (
                        axis in p._pair_spec.get("structure", {})
                        and bool(p._pair_spec.get("wrap"))
                    )
                )
                axis_obj.get_label().set_visible(show_axis_label)

                show_tick_labels = (
                    show_axis_label
                    or subplot_spec.get(f"share{axis}") not in (
                        True, "all", {"x": "col", "y": "row"}[axis]
                    )
                )
                for group in ("major", "minor"):
                    for t in getattr(axis_obj, f"get_{group}ticklabels")():
                        t.set_visible(show_tick_labels)

            # TODO we want right-side titles for row facets in most cases?
            # Let's have what we currently call "margin titles" but properly using the
            # ax.set_title interface (see my gist)
            title_parts = []
            for dim in ["col", "row"]:
                if sub[dim] is not None:
                    val = self._resolve_label(p, "title", f"{sub[dim]}")
                    if dim in p._labels:
                        key = self._resolve_label(p, dim, common.names.get(dim))
                        val = f"{key} {val}"
                    title_parts.append(val)

            has_col = sub["col"] is not None
            has_row = sub["row"] is not None
            show_title = (
                has_col and has_row
                or (has_col or has_row) and p._facet_spec.get("wrap")
                or (has_col and sub["top"])
                # TODO or has_row and sub["right"] and <right titles>
                or has_row  # TODO and not <right titles>
            )
            if title_parts:
                title = " | ".join(title_parts)
                title_text = ax.set_title(title)
                title_text.set_visible(show_title)
            elif not (has_col or has_row):
                title = self._resolve_label(p, "title", None)
                title_text = ax.set_title(title)

    def _compute_stats(self, spec: Plot, layers: list[Layer]) -> None:

        grouping_vars = [v for v in PROPERTIES if v not in "xy"]
        grouping_vars += ["col", "row", "group"]

        pair_vars = spec._pair_spec.get("structure", {})

        for layer in layers:

            data = layer["data"]
            mark = layer["mark"]
            stat = layer["stat"]

            if stat is None:
                continue

            iter_axes = itertools.product(*[
                pair_vars.get(axis, [axis]) for axis in "xy"
            ])

            old = data.frame

            if pair_vars:
                data.frames = {}
                data.frame = data.frame.iloc[:0]  # TODO to simplify typing

            for coord_vars in iter_axes:

                pairings = "xy", coord_vars

                df = old.copy()
                scales = self._scales.copy()

                for axis, var in zip(*pairings):
                    if axis != var:
                        df = df.rename(columns={var: axis})
                        drop_cols = [x for x in df if re.match(rf"{axis}\d+", str(x))]
                        df = df.drop(drop_cols, axis=1)
                        scales[axis] = scales[var]

                orient = layer["orient"] or mark._infer_orient(scales)

                if stat.group_by_orient:
                    grouper = [orient, *grouping_vars]
                else:
                    grouper = grouping_vars
                groupby = GroupBy(grouper)
                res = stat(df, groupby, orient, scales)

                if pair_vars:
                    data.frames[coord_vars] = res
                else:
                    data.frame = res

    def _get_scale(
        self, spec: Plot, var: str, prop: Property, values: Series
    ) -> Scale:

        if var in spec._scales:
            arg = spec._scales[var]
            if arg is None or isinstance(arg, Scale):
                scale = arg
            else:
                scale = prop.infer_scale(arg, values)
        else:
            scale = prop.default_scale(values)

        return scale

    def _get_subplot_data(self, df, var, view, share_state):

        if share_state in [True, "all"]:
            # The all-shared case is easiest, every subplot sees all the data
            seed_values = df[var]
        else:
            # Otherwise, we need to setup separate scales for different subplots
            if share_state in [False, "none"]:
                # Fully independent axes are also easy: use each subplot's data
                idx = self._get_subplot_index(df, view)
            elif share_state in df:
                # Sharing within row/col is more complicated
                use_rows = df[share_state] == view[share_state]
                idx = df.index[use_rows]
            else:
                # This configuration doesn't make much sense, but it's fine
                idx = df.index

            seed_values = df.loc[idx, var]

        return seed_values

    def _setup_scales(
        self, p: Plot,
        common: PlotData,
        layers: list[Layer],
        variables: list[str] | None = None,
    ) -> None:

        if variables is None:
            # Add variables that have data but not a scale, which happens
            # because this method can be called multiple time, to handle
            # variables added during the Stat transform.
            variables = []
            for layer in layers:
                variables.extend(layer["data"].frame.columns)
                for df in layer["data"].frames.values():
                    variables.extend(str(v) for v in df if v not in variables)
            variables = [v for v in variables if v not in self._scales]

        for var in variables:

            # Determine whether this is a coordinate variable
            # (i.e., x/y, paired x/y, or derivative such as xmax)
            m = re.match(r"^(?P<coord>(?P<axis>x|y)\d*).*", var)
            if m is None:
                coord = axis = None
            else:
                coord = m["coord"]
                axis = m["axis"]

            # Get keys that handle things like x0, xmax, properly where relevant
            prop_key = var if axis is None else axis
            scale_key = var if coord is None else coord

            if prop_key not in PROPERTIES:
                continue

            # Concatenate layers, using only the relevant coordinate and faceting vars,
            # This is unnecessarily wasteful, as layer data will often be redundant.
            # But figuring out the minimal amount we need is more complicated.
            cols = [var, "col", "row"]
            parts = [common.frame.filter(cols)]
            for layer in layers:
                parts.append(layer["data"].frame.filter(cols))
                for df in layer["data"].frames.values():
                    parts.append(df.filter(cols))
            var_df = pd.concat(parts, ignore_index=True)

            prop = PROPERTIES[prop_key]
            scale = self._get_scale(p, scale_key, prop, var_df[var])

            if scale_key not in p._variables:
                # TODO this implies that the variable was added by the stat
                # It allows downstream orientation inference to work properly.
                # But it feels rather hacky, so ideally revisit.
                scale._priority = 0  # type: ignore

            if axis is None:
                # We could think about having a broader concept of (un)shared properties
                # In general, not something you want to do (different scales in facets)
                # But could make sense e.g. with paired plots. Build later.
                share_state = None
                subplots = []
            else:
                share_state = self._subplots.subplot_spec[f"share{axis}"]
                subplots = [view for view in self._subplots if view[axis] == coord]

            # Shared categorical axes are broken on matplotlib<3.4.0.
            # https://github.com/matplotlib/matplotlib/pull/18308
            # This only affects us when sharing *paired* axes. This is a novel/niche
            # behavior, so we will raise rather than hack together a workaround.
            if axis is not None and Version(mpl.__version__) < Version("3.4.0"):
                paired_axis = axis in p._pair_spec.get("structure", {})
                cat_scale = isinstance(scale, Nominal)
                ok_dim = {"x": "col", "y": "row"}[axis]
                shared_axes = share_state not in [False, "none", ok_dim]
                if paired_axis and cat_scale and shared_axes:
                    err = "Sharing paired categorical axes requires matplotlib>=3.4.0"
                    raise RuntimeError(err)

            if scale is None:
                self._scales[var] = Scale._identity()
            else:
                try:
                    self._scales[var] = scale._setup(var_df[var], prop)
                except Exception as err:
                    raise PlotSpecError._during("Scale setup", var) from err

            if axis is None or (var != coord and coord in p._variables):
                # Everything below here applies only to coordinate variables
                continue

            # Set up an empty series to receive the transformed values.
            # We need this to handle piecemeal transforms of categories -> floats.
            transformed_data = []
            for layer in layers:
                index = layer["data"].frame.index
                empty_series = pd.Series(dtype=float, index=index, name=var)
                transformed_data.append(empty_series)

            for view in subplots:

                axis_obj = getattr(view["ax"], f"{axis}axis")
                seed_values = self._get_subplot_data(var_df, var, view, share_state)
                view_scale = scale._setup(seed_values, prop, axis=axis_obj)
                set_scale_obj(view["ax"], axis, view_scale._matplotlib_scale)

                for layer, new_series in zip(layers, transformed_data):
                    layer_df = layer["data"].frame
                    if var not in layer_df:
                        continue

                    idx = self._get_subplot_index(layer_df, view)
                    try:
                        new_series.loc[idx] = view_scale(layer_df.loc[idx, var])
                    except Exception as err:
                        spec_error = PlotSpecError._during("Scaling operation", var)
                        raise spec_error from err

            # Now the transformed data series are complete, set update the layer data
            for layer, new_series in zip(layers, transformed_data):
                layer_df = layer["data"].frame
                if var in layer_df:
                    layer_df[var] = new_series

    def _plot_layer(self, p: Plot, layer: Layer) -> None:

        data = layer["data"]
        mark = layer["mark"]
        move = layer["move"]

        default_grouping_vars = ["col", "row", "group"]  # TODO where best to define?
        grouping_properties = [v for v in PROPERTIES if v[0] not in "xy"]

        pair_variables = p._pair_spec.get("structure", {})

        for subplots, df, scales in self._generate_pairings(data, pair_variables):

            orient = layer["orient"] or mark._infer_orient(scales)

            def get_order(var):
                # Ignore order for x/y: they have been scaled to numeric indices,
                # so any original order is no longer valid. Default ordering rules
                # sorted unique numbers will correctly reconstruct intended order
                # TODO This is tricky, make sure we add some tests for this
                if var not in "xy" and var in scales:
                    return getattr(scales[var], "order", None)

            if orient in df:
                width = pd.Series(index=df.index, dtype=float)
                for view in subplots:
                    view_idx = self._get_subplot_data(
                        df, orient, view, p._shares.get(orient)
                    ).index
                    view_df = df.loc[view_idx]
                    if "width" in mark._mappable_props:
                        view_width = mark._resolve(view_df, "width", None)
                    elif "width" in df:
                        view_width = view_df["width"]
                    else:
                        view_width = 0.8  # TODO what default?
                    spacing = scales[orient]._spacing(view_df.loc[view_idx, orient])
                    width.loc[view_idx] = view_width * spacing
                df["width"] = width

            if "baseline" in mark._mappable_props:
                # TODO what marks should have this?
                # If we can set baseline with, e.g., Bar(), then the
                # "other" (e.g. y for x oriented bars) parameterization
                # is somewhat ambiguous.
                baseline = mark._resolve(df, "baseline", None)
            else:
                # TODO unlike width, we might not want to add baseline to data
                # if the mark doesn't use it. Practically, there is a concern about
                # Mark abstraction like Area / Ribbon
                baseline = 0 if "baseline" not in df else df["baseline"]
            df["baseline"] = baseline

            if move is not None:
                moves = move if isinstance(move, list) else [move]
                for move_step in moves:
                    move_by = getattr(move_step, "by", None)
                    if move_by is None:
                        move_by = grouping_properties
                    move_groupers = [*move_by, *default_grouping_vars]
                    if move_step.group_by_orient:
                        move_groupers.insert(0, orient)
                    order = {var: get_order(var) for var in move_groupers}
                    groupby = GroupBy(order)
                    df = move_step(df, groupby, orient, scales)

            df = self._unscale_coords(subplots, df, orient)

            grouping_vars = mark._grouping_props + default_grouping_vars
            split_generator = self._setup_split_generator(grouping_vars, df, subplots)

            mark._plot(split_generator, scales, orient)

        # TODO is this the right place for this?
        for view in self._subplots:
            view["ax"].autoscale_view()

        if layer["legend"]:
            self._update_legend_contents(p, mark, data, scales)

    def _unscale_coords(
        self, subplots: list[dict], df: DataFrame, orient: str,
    ) -> DataFrame:
        # TODO do we still have numbers in the variable name at this point?
        coord_cols = [c for c in df if re.match(r"^[xy]\D*$", str(c))]
        drop_cols = [*coord_cols, "width"] if "width" in df else coord_cols
        out_df = (
            df
            .drop(drop_cols, axis=1)
            .reindex(df.columns, axis=1)  # So unscaled columns retain their place
            .copy(deep=False)
        )

        for view in subplots:
            view_df = self._filter_subplot_data(df, view)
            axes_df = view_df[coord_cols]
            for var, values in axes_df.items():

                axis = getattr(view["ax"], f"{str(var)[0]}axis")
                # TODO see https://github.com/matplotlib/matplotlib/issues/22713
                transform = axis.get_transform().inverted().transform
                inverted = transform(values)
                out_df.loc[values.index, str(var)] = inverted

                if var == orient and "width" in view_df:
                    width = view_df["width"]
                    out_df.loc[values.index, "width"] = (
                        transform(values + width / 2) - transform(values - width / 2)
                    )

        return out_df

    def _generate_pairings(
        self, data: PlotData, pair_variables: dict,
    ) -> Generator[
        tuple[list[dict], DataFrame, dict[str, Scale]], None, None
    ]:
        # TODO retype return with subplot_spec or similar

        iter_axes = itertools.product(*[
            pair_variables.get(axis, [axis]) for axis in "xy"
        ])

        for x, y in iter_axes:

            subplots = []
            for view in self._subplots:
                if (view["x"] == x) and (view["y"] == y):
                    subplots.append(view)

            if data.frame.empty and data.frames:
                out_df = data.frames[(x, y)].copy()
            elif not pair_variables:
                out_df = data.frame.copy()
            else:
                if data.frame.empty and data.frames:
                    out_df = data.frames[(x, y)].copy()
                else:
                    out_df = data.frame.copy()

            scales = self._scales.copy()
            if x in out_df:
                scales["x"] = self._scales[x]
            if y in out_df:
                scales["y"] = self._scales[y]

            for axis, var in zip("xy", (x, y)):
                if axis != var:
                    out_df = out_df.rename(columns={var: axis})
                    cols = [col for col in out_df if re.match(rf"{axis}\d+", str(col))]
                    out_df = out_df.drop(cols, axis=1)

            yield subplots, out_df, scales

    def _get_subplot_index(self, df: DataFrame, subplot: dict) -> Index:

        dims = df.columns.intersection(["col", "row"])
        if dims.empty:
            return df.index

        keep_rows = pd.Series(True, df.index, dtype=bool)
        for dim in dims:
            keep_rows &= df[dim] == subplot[dim]
        return df.index[keep_rows]

    def _filter_subplot_data(self, df: DataFrame, subplot: dict) -> DataFrame:
        # TODO note redundancies with preceding function ... needs refactoring
        dims = df.columns.intersection(["col", "row"])
        if dims.empty:
            return df

        keep_rows = pd.Series(True, df.index, dtype=bool)
        for dim in dims:
            keep_rows &= df[dim] == subplot[dim]
        return df[keep_rows]

    def _setup_split_generator(
        self, grouping_vars: list[str], df: DataFrame, subplots: list[dict[str, Any]],
    ) -> Callable[[], Generator]:

        grouping_keys = []
        grouping_vars = [
            v for v in grouping_vars if v in df and v not in ["col", "row"]
        ]
        for var in grouping_vars:
            order = getattr(self._scales[var], "order", None)
            if order is None:
                order = categorical_order(df[var])
            grouping_keys.append(order)

        def split_generator(keep_na=False) -> Generator:

            for view in subplots:

                axes_df = self._filter_subplot_data(df, view)

                with pd.option_context("mode.use_inf_as_na", True):
                    if keep_na:
                        # The simpler thing to do would be x.dropna().reindex(x.index).
                        # But that doesn't work with the way that the subset iteration
                        # is written below, which assumes data for grouping vars.
                        # Matplotlib (usually?) masks nan data, so this should "work".
                        # Downstream code can also drop these rows, at some speed cost.
                        present = axes_df.notna().all(axis=1)
                        nulled = {}
                        for axis in "xy":
                            if axis in axes_df:
                                nulled[axis] = axes_df[axis].where(present)
                        axes_df = axes_df.assign(**nulled)
                    else:
                        axes_df = axes_df.dropna()

                subplot_keys = {}
                for dim in ["col", "row"]:
                    if view[dim] is not None:
                        subplot_keys[dim] = view[dim]

                if not grouping_vars or not any(grouping_keys):
                    if not axes_df.empty:
                        yield subplot_keys, axes_df.copy(), view["ax"]
                    continue

                grouped_df = axes_df.groupby(grouping_vars, sort=False, as_index=False)

                for key in itertools.product(*grouping_keys):

                    # Pandas fails with singleton tuple inputs
                    pd_key = key[0] if len(key) == 1 else key

                    try:
                        df_subset = grouped_df.get_group(pd_key)
                    except KeyError:
                        # TODO (from initial work on categorical plots refactor)
                        # We are adding this to allow backwards compatability
                        # with the empty artists that old categorical plots would
                        # add (before 0.12), which we may decide to break, in which
                        # case this option could be removed
                        df_subset = axes_df.loc[[]]

                    if df_subset.empty:
                        continue

                    sub_vars = dict(zip(grouping_vars, key))
                    sub_vars.update(subplot_keys)

                    # TODO need copy(deep=...) policy (here, above, anywhere else?)
                    yield sub_vars, df_subset.copy(), view["ax"]

        return split_generator

    def _update_legend_contents(
        self,
        p: Plot,
        mark: Mark,
        data: PlotData,
        scales: dict[str, Scale],
    ) -> None:
        """Add legend artists / labels for one layer in the plot."""
        if data.frame.empty and data.frames:
            legend_vars: list[str] = []
            for frame in data.frames.values():
                frame_vars = frame.columns.intersection(list(scales))
                legend_vars.extend(v for v in frame_vars if v not in legend_vars)
        else:
            legend_vars = list(data.frame.columns.intersection(list(scales)))

        # First pass: Identify the values that will be shown for each variable
        schema: list[tuple[
            tuple[str, str | int], list[str], tuple[list, list[str]]
        ]] = []
        schema = []
        for var in legend_vars:
            var_legend = scales[var]._legend
            if var_legend is not None:
                values, labels = var_legend
                for (_, part_id), part_vars, _ in schema:
                    if data.ids[var] == part_id:
                        # Allow multiple plot semantics to represent same data variable
                        part_vars.append(var)
                        break
                else:
                    title = self._resolve_label(p, var, data.names[var])
                    entry = (title, data.ids[var]), [var], (values, labels)
                    schema.append(entry)

        # Second pass, generate an artist corresponding to each value
        contents: list[tuple[tuple[str, str | int], Any, list[str]]] = []
        for key, variables, (values, labels) in schema:
            artists = []
            for val in values:
                artist = mark._legend_artist(variables, val, scales)
                if artist is not None:
                    artists.append(artist)
            if artists:
                contents.append((key, artists, labels))

        self._legend_contents.extend(contents)

    def _make_legend(self, p: Plot) -> None:
        """Create the legend artist(s) and add onto the figure."""
        # Combine artists representing same information across layers
        # Input list has an entry for each distinct variable in each layer
        # Output dict has an entry for each distinct variable
        merged_contents: dict[
            tuple[str, str | int], tuple[list[Artist], list[str]],
        ] = {}
        for key, new_artists, labels in self._legend_contents:
            # Key is (name, id); we need the id to resolve variable uniqueness,
            # but will need the name in the next step to title the legend
            if key in merged_contents:
                # Copy so inplace updates don't propagate back to legend_contents
                existing_artists = merged_contents[key][0]
                for i, artist in enumerate(existing_artists):
                    # Matplotlib accepts a tuple of artists and will overlay them
                    if isinstance(artist, tuple):
                        artist += new_artists[i],
                    else:
                        existing_artists[i] = artist, new_artists[i]
            else:
                merged_contents[key] = new_artists.copy(), labels

        # TODO explain
        loc = "center right" if self._pyplot else "center left"

        base_legend = None
        for (name, _), (handles, labels) in merged_contents.items():

            legend = mpl.legend.Legend(
                self._figure,
                handles,
                labels,
                title=name,
                loc=loc,
                bbox_to_anchor=(.98, .55),
            )

            if base_legend:
                # Matplotlib has no public API for this so it is a bit of a hack.
                # Ideally we'd define our own legend class with more flexibility,
                # but that is a lot of work!
                base_legend_box = base_legend.get_children()[0]
                this_legend_box = legend.get_children()[0]
                base_legend_box.get_children().extend(this_legend_box.get_children())
            else:
                base_legend = legend
                self._figure.legends.append(legend)

    def _finalize_figure(self, p: Plot) -> None:

        for sub in self._subplots:
            ax = sub["ax"]
            for axis in "xy":
                axis_key = sub[axis]
                axis_obj = getattr(ax, f"{axis}axis")

                # Axis limits
                if axis_key in p._limits:
                    convert_units = getattr(ax, f"{axis}axis").convert_units
                    a, b = p._limits[axis_key]
                    lo = a if a is None else convert_units(a)
                    hi = b if b is None else convert_units(b)
                    if isinstance(a, str):
                        lo = cast(float, lo) - 0.5
                    if isinstance(b, str):
                        hi = cast(float, hi) + 0.5
                    ax.set(**{f"{axis}lim": (lo, hi)})

                # Nominal scale special-casing
                if isinstance(self._scales.get(axis_key), Nominal):
                    axis_obj.grid(False, which="both")
                    if axis_key not in p._limits:
                        nticks = len(axis_obj.get_major_ticks())
                        lo, hi = -.5, nticks - .5
                        if axis == "y":
                            lo, hi = hi, lo
                        set_lim = getattr(ax, f"set_{axis}lim")
                        set_lim(lo, hi, auto=None)

        engine_default = None if p._target is not None else "tight"
        layout_engine = p._layout_spec.get("engine", engine_default)
        set_layout_engine(self._figure, layout_engine)
