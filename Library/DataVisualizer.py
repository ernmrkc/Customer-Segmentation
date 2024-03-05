from typing import Optional, Tuple
from tabulate import tabulate
from enum import Enum
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

class MarkerType(Enum):
    DOT = "."                   # Dot
    CIRCLE = "o"                # Circle
    CIRCLE_BOLD = "8"           # Circle Bold
    TRIANGLE_DOWN = "v"         # Downward Triangle
    TRIANGLE_UP = "^"           # Upward Triangle
    TRIANGLE_LEFT = "<"         # Leftward Triangle
    TRIANGLE_RIGHT = ">"        # Rightward Triangle
    SQUARE = "s"                # Square
    PLUS = "p"                  # Plus
    PLUS_BOLD = "P"             # Bold Plus
    STAR = "*"                  # Star
    STAR_BOLD = "X"             # Bold Star
    HEXAGON = "h"               # Hexagon
    HEXAGON_BOLD = "H"          # Bold Hexagon
    DIAMOND = "d"               # Diamond
    DIAMOND_BOLD = "D"          # Bold Diamond
    
    
class LineStyle(Enum):
    SOLID = "-"                 # Solid Line
    DASHED = "--"               # Dashed Line
    DASHDOT = "-."              # Dash-Dotted Line
    DOTTED = ":"                # Dotted Line
    NOTHING = ""                # Draw Nothing
    
    
class ColorSingle(Enum):
    BLUE = "b"                  # Blue
    GREEN = "g"                 # Green
    RED = "r"                   # Red
    CYAN = "c"                  # Cyan
    MAGENTA = "m"               # Magenta
    YELLOW = "y"                # Yellow
    BLACK = "k"                 # Black
    WHITE = "w"                 # White
    
  
class ColorPalette(Enum):
    ACCENT = "Accent"                           # An accent palette with bright colors.
    ACCENT_R = "Accent_r"                       # Reverse of the Accent palette.
    BLUES = "Blues"                             # A sequential palette of varying shades of blue.
    BLUES_R = "Blues_r"                         # Reverse of the Blues palette.
    BRBG = "BrBG"                               # A diverging palette with brown to blue-green.
    BRBG_R = "BrBG_r"                           # Reverse of the BrBG palette.
    BUGN = "BuGn"                               # A sequential palette from light blue to green.
    BUGN_R = "BuGn_r"                           # Reverse of the BuGn palette.
    BUPU = "BuPu"                               # A sequential palette from light blue to purple.
    BUPU_R = "BuPu_r"                           # Reverse of the BuPu palette.
    CMRMAP = "CMRmap"                           # A sequential palette designed for scientific data, ranging from dark to light.
    CMRMAP_R = "CMRmap_r"                       # Reverse of the CMRmap palette.
    DARK2 = "Dark2"                             # A qualitative palette with dark colors.
    DARK2_R = "Dark2_r"                         # Reverse of the Dark2 palette.
    GNBU = "GnBu"                               # A sequential palette from light green to blue.
    GNBU_R = "GnBu_r"                           # Reverse of the GnBu palette.
    GRAYS = "Grays"                             # A sequential grayscale palette, from light to dark.
    GREENS = "Greens"                           # A sequential palette of varying shades of green.
    GREENS_R = "Greens_r"                       # Reverse of the Greens palette.
    GREYS = "Greys"                             # Another option for a sequential grayscale palette.
    GREYS_R = "Greys_r"                         # Reverse of the Greys palette.
    ORRD = "OrRd"                               # A sequential palette from light orange to dark red.
    ORRD_R = "OrRd_r"                           # Reverse of the OrRd palette.
    ORANGES = "Oranges"                         # A sequential palette of varying shades of orange.
    ORANGES_R = "Oranges_r"                     # Reverse of the Oranges palette.
    PRGN = "PRGn"                               # A diverging palette with purple to green.
    PRGN_R = "PRGn_r"                           # Reverse of the PRGn palette.
    PAIRED = "Paired"                           # A qualitative palette with paired colors.
    PAIRED_R = "Paired_r"                       # Reverse of the Paired palette.
    PASTEL1 = "Pastel1"                         # A qualitative palette with pastel colors.
    PASTEL1_R = "Pastel1_r"                     # Reverse of the Pastel1 palette.
    PASTEL2 = "Pastel2"                         # Another qualitative palette with pastel colors.
    PASTEL2_R = "Pastel2_r"                     # Reverse of the Pastel2 palette.
    PIYG = "PiYG"                               # A diverging palette with pink to green.
    PIYG_R = "PiYG_r"                           # Reverse of the PiYG palette.
    PUBU = "PuBu"                               # A sequential palette from light purple to blue.
    PUBUGN = "PuBuGn"                           # A sequential palette from light purple to blue-green.
    PUBUGN_R = "PuBuGn_r"                       # Reverse of the PuBuGn palette.
    PUBU_R = "PuBu_r"                           # Reverse of the PuBu palette.
    PUOR = "PuOr"                               # A diverging palette with orange to purple.
    PUOR_R = "PuOr_r"                           # Reverse of the PuOr palette.
    PURD = "PuRd"                               # A sequential palette from light purple to dark red.
    PURD_R = "PuRd_r"                           # Reverse of the PuRd palette.
    PURPLES = "Purples"                         # A sequential palette of varying shades of purple.
    PURPLES_R = "Purples_r"                     # Reverse of the Purples palette.
    RDBU = "RdBu"                               # A diverging palette with red to blue.
    RDBU_R = "RdBu_r"                           # Reverse of the RdBu palette.
    RDGY = "RdGy"                               # A diverging palette with red to gray.
    RDGY_R = "RdGy_r"                           # Reverse of the RdGy palette.
    RDPU = "RdPu"                               # A sequential palette from light red to purple.
    RDPU_R = "RdPu_r"                           # Reverse of the RdPu palette.
    RDYLBU = "RdYlBu"                           # A diverging palette with red, yellow to blue.
    RDYLBU_R = "RdYlBu_r"                       # Reverse of the RdYlBu palette.
    RDYLGN = "RdYlGn"                           # A diverging palette with red, yellow to green.
    RDYLGN_R = "RdYlGn_r"                       # Reverse of the RdYlGn palette.
    REDS = "Reds"                               # A sequential palette of varying shades of red.
    REDS_R = "Reds_r"                           # Reverse of the Reds palette.
    SET1 = "Set1"                               # A qualitative palette with distinct colors.
    SET1_R = "Set1_r"                           # Reverse of the Set1 palette.
    SET2 = "Set2"                               # Another qualitative palette with distinct, slightly muted colors.
    SET2_R = "Set2_r"                           # Reverse of the Set2 palette.
    SET3 = "Set3"                               # Another qualitative palette with distinct colors.
    SET3_R = "Set3_r"                           # Reverse of the Set3 palette.
    SPECTRAL = "Spectral"                       # A diverging palette with a spectral progression.
    SPECTRAL_R = "Spectral_r"                   # Reverse of the Spectral palette.
    WISTIA = "Wistia"                           # A sequential palette with a light tint of yellow.
    WISTIA_R = "Wistia_r"                       # Reverse of the Wistia palette.
    YLGN = "YlGn"                               # A sequential palette from light yellow to green.
    YLGNBU = "YlGnBu"                           # A sequential palette from light yellow, green to blue.
    YLGNBU_R = "YlGnBu_r"                       # Reverse of the YlGnBu palette.
    YLGN_R = "YlGn_r"                           # Reverse of the YlGn palette.
    YLORBR = "YlOrBr"                           # A sequential palette from light yellow, orange to brown.
    YLORBR_R = "YlOrBr_r"                       # Reverse of the YlOrBr palette.
    YLORRD = "YlOrRd"                           # A sequential palette from light yellow, orange to red.
    YLORRD_R = "YlOrRd_r"                       # Reverse of the YlOrRd palette.
    AFMHOT = "afmhot"                           # A sequential palette with a hot, fire-like progression.
    AFMHOT_R = "afmhot_r"                       # Reverse of the afmhot palette.
    AUTUMN = "autumn"                           # A sequential palette with shades of autumn colors.
    AUTUMN_R = "autumn_r"                       # Reverse of the autumn palette.
    BINARY = "binary"                           # A binary palette with black and white.
    BINARY_R = "binary_r"                       # Reverse of the binary palette.
    BONE = "bone"                               # A grayscale palette with a hint of blue, resembling the appearance of a bone X-ray.
    BONE_R = "bone_r"                           # Reverse of the bone palette.
    BRG = "brg"                                 # A palette transitioning between blue, red, and green.
    BRG_R = "brg_r"                             # Reverse of the brg palette.
    BWR = "bwr"                                 # A diverging palette with blue and red.
    BWR_R = "bwr_r"                             # Reverse of the bwr palette.
    CIVIDIS = "cividis"                         # A perceptually uniform palette designed to be accessible to viewers with color vision deficiencies.
    CIVIDIS_R = "cividis_r"                     # Reverse of the cividis palette.
    COOL = "cool"                               # A sequential palette with shades of cyan and magenta.
    COOL_R = "cool_r"                           # Reverse of the cool palette.
    COOLWARM = "coolwarm"                       # A diverging palette with cool and warm tones.
    COOLWARM_R = "coolwarm_r"                   # Reverse of the coolwarm palette.
    COPPER = "copper"                           # A sequential palette with copper tones.
    COPPER_R = "copper_r"                       # Reverse of the copper palette.
    CUBEHELIX = "cubehelix"                     # A palette generated with a helical trajectory in the RGB color space, designed to be perceptually uniform.
    CUBEHELIX_R = "cubehelix_r"                 # Reverse of the cubehelix palette.
    FLAG = "flag"                               # A patterned palette with a flag-like appearance.
    FLAG_R = "flag_r"                           # Reverse of the flag palette.
    GIST_EARTH = "gist_earth"                   # A palette designed to represent elevation with earth-like colors.
    GIST_EARTH_R = "gist_earth_r"               # Reverse of the gist_earth palette.
    GIST_GRAY = "gist_gray"                     # A grayscale palette designed for scientific visualization.
    GIST_GRAY_R = "gist_gray_r"                 # Reverse of the gist_gray palette.
    GIST_GREY = "gist_grey"                     # Another name for the gist_gray palette.
    GIST_HEAT = "gist_heat"                     # A palette with a heat-like progression, suitable for heatmaps.
    GIST_HEAT_R = "gist_heat_r"                 # Reverse of the gist_heat palette.
    GIST_NCAR = "gist_ncar"                     # A palette based on the National Center for Atmospheric Research's color table.
    GIST_NCAR_R = "gist_ncar_r"                 # Reverse of the gist_ncar palette.
    GIST_RAINBOW = "gist_rainbow"               # A rainbow palette designed for scientific visualization.
    GIST_RAINBOW_R = "gist_rainbow_r"           # Reverse of the gist_rainbow palette.
    GIST_STERN = "gist_stern"                   # A palette with a stern-like progression, including both cool and warm tones.
    GIST_STERN_R = "gist_stern_r"               # Reverse of the gist_stern palette.
    GIST_YARG = "gist_yarg"                     # A grayscale palette with a progression from dark to light (the name "yarg" is "gray" spelled backwards).
    GIST_YARG_R = "gist_yarg_r"                 # Reverse of the gist_yarg palette.
    GIST_YERG = "gist_yerg"                     # A palette with a progression from yellow to green to red.
    GNUPLOT = "gnuplot"                         # A palette inspired by the default color table of the Gnuplot plotting utility.
    GNUPLOT2 = "gnuplot2"                       # Another palette inspired by Gnuplot, with a different color progression.
    GNUPLOT2_R = "gnuplot2_r"                   # Reverse of the gnuplot2 palette.
    GNUPLOT_R = "gnuplot_r"                     # Reverse of the gnuplot palette.
    GRAY = "gray"                               # A grayscale palette, from black to white.
    GRAY_R = "gray_r"                           # Reverse of the gray palette.
    GREY = "grey"                               # Another name for the gray palette.
    HOT = "hot"                                 # A sequential palette with a hot, fire-like progression.
    HOT_R = "hot_r"                             # Reverse of the hot palette.
    HSV = "hsv"                                 # A palette with hues arranged in a circular progression, based on the HSV color space.
    HSV_R = "hsv_r"                             # Reverse of the hsv palette.
    INFERNO = "inferno"                         # A perceptually uniform palette with a fiery appearance.
    INFERNO_R = "inferno_r"                     # Reverse of the inferno palette.
    JET = "jet"                                 # A rainbow palette that transitions through most of the visible spectrum.
    JET_R = "jet_r"                             # Reverse of the jet palette.
    MAGMA = "magma"                             # A perceptually uniform palette with dark, rich colors.
    MAGMA_R = "magma_r"                         # Reverse of the magma palette.
    NIPY_SPECTRAL = "nipy_spectral"             # A palette with a wide range of colors, designed for scientific visualization.
    NIPY_SPECTRAL_R = "nipy_spectral_r"         # Reverse of the nipy_spectral palette.
    OCEAN = "ocean"                             # A palette with blue tones, resembling the ocean.
    OCEAN_R = "ocean_r"                         # Reverse of the ocean palette.
    PINK = "pink"                               # A sequential palette with pink tones.
    PINK_R = "pink_r"                           # Reverse of the pink palette.
    PLASMA = "plasma"                           # A perceptually uniform palette with vibrant colors.
    PLASMA_R = "plasma_r"                       # Reverse of the plasma palette.
    PRISM = "prism"                             # A patterned palette with a prismatic appearance.
    PRISM_R = "prism_r"                         # Reverse of the prism palette.
    RAINBOW = "rainbow"                         # A rainbow palette with a wide range of colors.
    RAINBOW_R = "rainbow_r"                     # Reverse of the rainbow palette.
    SEISMIC = "seismic"                         # A diverging palette with red to blue, suitable for highlighting differences in data.
    SEISMIC_R = "seismic_r"                     # Reverse of the seismic palette.
    SPRING = "spring"                           # A sequential palette with shades of spring colors.
    SPRING_R = "spring_r"                       # Reverse of the spring palette.
    SUMMER = "summer"                           # A sequential palette with summer colors, typically green to yellow.
    SUMMER_R = "summer_r"                       # Reverse of the summer palette.
    TAB10 = "tab10"                             # A qualitative palette with ten colors, designed for categorical data.
    TAB10_R = "tab10_r"                         # Reverse of the tab10 palette.
    TAB20 = "tab20"                             # A qualitative palette with twenty colors, designed for differentiating a wide range of categories.
    TAB20_R = "tab20_r"                         # Reverse of the tab20 palette.
    TAB20B = "tab20b"                           # Another qualitative palette with twenty colors, providing additional options for categorization.
    TAB20B_R = "tab20b_r"                       # Reverse of the tab20b palette.
    TAB20C = "tab20c"                           # Another qualitative palette with twenty colors, offering even more variety for categorical data.
    TAB20C_R = "tab20c_r"                       # Reverse of the tab20c palette.
    TERRAIN = "terrain"                         # A palette designed to represent topographical elevation, with green, brown, and white colors.
    TERRAIN_R = "terrain_r"                     # Reverse of the terrain palette.
    TURBO = "turbo"                             # A high-contrast palette with a wide range of colors, designed for clarity in data visualization.
    TURBO_R = "turbo_r"                         # Reverse of the turbo palette.
    TWILIGHT = "twilight"                       # A cyclic palette with a twilight theme, transitioning through a series of cool and warm colors.
    TWILIGHT_R = "twilight_r"                   # Reverse of the twilight palette.
    TWILIGHT_SHIFTED = "twilight_shifted"       # A variation of the twilight palette with shifted color values.
    TWILIGHT_SHIFTED_R = "twilight_shifted_r"   # Reverse of the twilight_shifted palette.
    VIRIDIS = "viridis"                         # A perceptually uniform palette with green-blue-violet colors.
    VIRIDIS_R = "viridis_r"                     # Reverse of the viridis palette.
    WINTER = "winter"                           # A sequential palette with cool, winter-themed colors.
    WINTER_R = "winter_r"                       # Reverse of the winter palette.
    PASTEL = "pastel"                           # A palette with soft, muted colors, suitable for creating gentle and aesthetically pleasing visualizations.
    HUSL = "husl"                               # A color palette based on the HUSL color space (Hue, Saturation, Lightness), designed for better perception and aesthetics, making it particularly useful for color choices in data visualization.
    

class DataVisualizer:
    def __init__(self):
        """
        Initializes the DataPreprocessor class.
        """
        pass   
    
    def plotColumnChart(self, dataframe: pd.DataFrame, column1: str, column2: Optional[str] = None, chart_type: str = "Histogram", figSizeW: int = 10, figSizeH: int = 6) -> None:
        """
        Plots a specified chart type for the given columns of a DataFrame.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame containing the data.
        - column1 (str): The name of the primary column for the chart.
        - column2 (Optional[str]): The name of the secondary column for the chart (required for Scatter and Box plots).
        - chart_type (str): The type of chart to plot ('Histogram', 'Scatter', 'Box', 'Countplot'). Default is 'Histogram'.
        - figSizeW (int): The width of the figure. Default is 10.
        - figSizeH (int): The height of the figure. Default is 6.

        Returns:
        - None: The function plots the chart and returns None.
        """
        plt.figure(figsize=(figSizeW, figSizeH))

        if chart_type == "Histogram":
            plt.hist(dataframe[column1])
            plt.xlabel(column1)
        elif chart_type == "Scatter":
            if column2:
                plt.scatter(dataframe[column1], dataframe[column2])
                plt.xlabel(column1)
                plt.ylabel(column2)
            else:
                raise ValueError("A second column must be provided for Scatter plots.")
        elif chart_type == "Box":
            if column2:
                sns.boxplot(x=dataframe[column1], y=dataframe[column2])
            else:
                sns.boxplot(x=dataframe[column1])
            plt.xlabel(column1)
            plt.ylabel(column2 if column2 else "")
        elif chart_type == "Countplot":
            sns.countplot(x=dataframe[column1])
            plt.xlabel(column1)
        else:
            raise ValueError(f"'{chart_type}' is not a supported chart type. Supported types: Histogram, Scatter, Box, Countplot.")

        plt.xticks(rotation=90)
        plt.show()
        
    def plotCorrelationHeatmap(self, dataframe: pd.DataFrame, figSizeW: int = 10, figSizeH: int = 10) -> Tuple[plt.Figure, None]:
        """
        Plots a correlation heatmap for the numerical columns of a DataFrame and lists non-numerical columns that were not included.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to analyze.
        - figSizeW (int): The width of the heatmap figure. Default is 10.
        - figSizeH (int): The height of the heatmap figure. Default is 10.

        Returns:
        - Tuple[plt.Figure, None]: A tuple containing the matplotlib Figure object of the heatmap and None.
        """
        # Identify non-numeric columns
        non_numeric_columns = dataframe.select_dtypes(exclude=['number']).columns.tolist()

        # Print non-numeric columns
        if non_numeric_columns:
            print("Non-numeric columns not included in the correlation heatmap:")
            print(tabulate([[col] for col in non_numeric_columns], headers=["Column Name"]))
        else:
            print("All columns are numeric and included in the heatmap.")

        # Plot heatmap
        plt.figure(figsize=(figSizeW, figSizeH))
        heatmap_fig = sns.heatmap(dataframe.corr(), annot=True, vmin=-1, vmax=1, center=0)
        plt.show()

        return heatmap_fig, None

    def draw_line_chart(self, 
        data1: List[Union[int, float]],
        data2: List[Union[int, float]],
        marker: Optional[MarkerType] = MarkerType.CIRCLE,
        linestyle: Optional[LineStyle] = LineStyle.SOLID,
        color: Optional[ColorSingle] = ColorSingle.BLUE,
        chart_title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        legend_title: str = "",
        show_grid: bool = True,
        figure_height: int = 10,
        figure_width: int = 10
    ) -> None:
        """
        Draws a line chart with the given datasets and visual styles.

        Parameters:
        - data1, data2: Lists of data points for the x and y axes.
        - marker: Enum value for the marker style.
        - linestyle: Enum value for the line style.
        - color: Enum value for the line color.
        - chart_title: Title of the chart.
        - xlabel, ylabel: Labels for the x and y axes.
        - legend_title: Title for the chart legend.
        - show_grid: Whether to display a grid.
        - figure_height, figure_width: Dimensions of the chart.
        """
        plt.figure(figsize=(figure_width, figure_height))
        plt.plot(data1, data2, marker=marker.value, linestyle=linestyle.value, color=color.value, label=legend_title)
        plt.title(chart_title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(show_grid)
        if legend_title:
            plt.legend()
        plt.show()
        
    def draw_scatter_chart(self, 
        data1: List[Union[int, float]],
        data2: List[Union[int, float]],
        marker: Optional[MarkerType] = MarkerType.CIRCLE,
        color: Optional[ColorSingle] = ColorSingle.BLUE,
        chart_title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        legend_title: str = "",
        show_grid: bool = True,
        figure_height: int = 10,
        figure_width: int = 10
    ) -> None:
        """
        Draws a scatter chart with the given datasets and visual styles.

        Parameters:
        - data1, data2: Lists of data points for the x and y axes.
        - marker: Enum value for the marker style.
        - color: Enum value for the chart color.
        - chart_title: Title of the chart.
        - xlabel, ylabel: Labels for the x and y axes.
        - legend_title: Title for the chart legend.
        - show_grid: Whether to display a grid.
        - figure_height, figure_width: Dimensions of the chart.
        """
        plt.figure(figsize=(figure_width, figure_height))
        plt.scatter(data1, data2, marker=marker.value, color=color.value, label=legend_title)
        plt.title(chart_title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(show_grid)
        if legend_title:
            plt.legend()
        plt.show()

    def draw_bar_chart(self,
        categories: List[str],
        values: List[Union[int, float]],
        color: Optional[ColorSingle] = ColorSingle.BLUE,
        chart_title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        legend_title: str = "",
        show_grid: bool = True,
        figure_height: int = 10,
        figure_width: int = 10
    ) -> None:
        """
        Draws a bar chart for the given data.

        Parameters:
        - categories: List of categories for the x-axis.
        - values: List of values for each category.
        - color: Enum value for the bars' color.
        - chart_title: Title of the chart.
        - xlabel, ylabel: Labels for the x and y axes.
        - legend_title: Title for the chart legend.
        - show_grid: Whether to display a grid.
        - figure_height, figure_width: Dimensions of the chart.
        """
        plt.figure(figsize=(figure_width, figure_height))
        plt.bar(categories, values, color=color.value, label=legend_title)
        plt.title(chart_title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(show_grid, axis='y')
        if legend_title:
            plt.legend()
        plt.show()
        
    def draw_pie_chart(self,
        data: List[float],
        labels: List[str],
        color_palette: Optional[ColorPalette] = ColorPalette.BLUES,
        chart_title: str = "",
        figure_height: int = 10,
        figure_width: int = 10
    ) -> None:
        """
        Draws a pie chart for the given data.

        Parameters:
        - data: List of numerical data points for the pie slices.
        - labels: List of labels for each pie slice.
        - color_palette: Enum value for the chart's color palette.
        - chart_title: Title of the chart.
        - figure_height, figure_width: Dimensions of the chart.
        """
        # Generate a color palette with seaborn
        sns.set_palette(sns.color_palette(color_palette.value))
        colors = sns.color_palette(None, len(data))  # None uses the current palette

        plt.figure(figsize=(figure_width, figure_height))
        plt.pie(data, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        plt.title(chart_title)
        plt.show()
        
    def draw_histogram_chart(self,
        data: List[float],
        bin_count: int,
        fill_color: Optional[ColorSingle] = ColorSingle.BLUE,
        opacity: float = 1.0,
        edge_color: Optional[ColorSingle] = ColorSingle.BLACK,
        chart_title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        show_grid: bool = True,
        figure_height: int = 10,
        figure_width: int = 10
    ) -> None:
        """
        Draws a histogram for the given data.

        Parameters:
        - data: List of numerical data points.
        - bin_count: Number of bins (or regions) to divide the data into.
        - fill_color: Enum value for the bars' fill color.
        - opacity: Opacity of the bars, ranging from 0 to 1.
        - edge_color: Enum value for the bars' edge color.
        - chart_title: Title of the chart.
        - xlabel, ylabel: Labels for the x and y axes.
        - show_grid: Whether to display a grid.
        - figure_height, figure_width: Dimensions of the chart.
        """
        # Validate opacity
        if not 0 <= opacity <= 1:
            print("The opacity value must be between 0 and 1.")
            return

        plt.figure(figsize=(figure_width, figure_height))
        plt.hist(data, bins=bin_count, color=fill_color.value, alpha=opacity, edgecolor=edge_color.value)
        plt.title(chart_title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(show_grid, axis='y')
        plt.show()
        
    def draw_3d_surface_chart(self,
        dataX: np.ndarray,
        dataY: np.ndarray,
        dataZ: np.ndarray,
        color_palette: Optional[ColorPalette] = ColorPalette.COOL,
        chart_title: str = "",
        figure_height: int = 10,
        figure_width: int = 10
    ) -> None:
        """
        Draws a 3D surface chart for the given data.

        Parameters:
        - dataX, dataY, dataZ: Numpy arrays of the x, y, and z coordinates.
        - color_palette: Enum value for the chart's color palette.
        - chart_title: Title of the chart.
        - figure_height, figure_width: Dimensions of the chart.
        """
        fig = plt.figure(figsize=(figure_width, figure_height))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(dataX, dataY, dataZ, cmap=color_palette.value)
        ax.set_title(chart_title)
        fig.colorbar(surf, shrink=0.5, aspect=5)  # Optional: Adds a color bar
        plt.show()
        
    def draw_area_chart(self, 
        data: List[float],
        y1: List[float],
        y2: List[float],
        colorY1: Optional[ColorSingle] = ColorSingle.RED,
        colorY2: Optional[ColorSingle] = ColorSingle.BLUE,
        labelY1: str = "",
        labelY2: str = "",
        chart_title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        show_grid: bool = True,
        figure_height: int = 10,
        figure_width: int = 10
    ) -> None:
        """
        Draws an area chart for the given datasets.

        Parameters:
        - data: X-axis values.
        - y1, y2: Y-axis values for the two datasets to be filled.
        - colorY1, colorY2: ColorSingle enum values for the fill colors of y1 and y2.
        - labelY1, labelY2: Labels for the y1 and y2 datasets.
        - chart_title: Title of the chart.
        - xlabel, ylabel: Labels for the X and Y axes.
        - show_grid: Whether to display a grid.
        - figure_height, figure_width: Dimensions of the chart.
        """
        plt.figure(figsize=(figure_width, figure_height))
        plt.fill_between(data, y1, color=colorY1.value, alpha=0.5, label=labelY1)
        plt.fill_between(data, y2, color=colorY2.value, alpha=0.5, label=labelY2)
        plt.title(chart_title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(show_grid)
        if labelY1 or labelY2:
            plt.legend()
        plt.show()
        
    def draw_radar_chart(self,
        data: List[float],
        labels: List[str],
        color: ColorSingle = ColorSingle.BLUE,
        opacity: float = 0.5,
        chart_title: str = "",
        figure_height: int = 10,
        figure_width: int = 10
    ) -> None:
        """
        Draws a radar chart for the given dataset.

        Parameters:
        - data: Numerical data points for each segment of the radar chart.
        - labels: Labels for each segment of the radar chart.
        - color: ColorSingle enum value for the fill color of the radar chart.
        - opacity: Opacity of the fill color, ranging from 0 to 1.
        - chart_title: Title of the chart.
        - figure_height, figure_width: Dimensions of the chart.
        """
        # Ensure data and labels are aligned
        if len(data) != len(labels):
            raise ValueError("Data and labels must have the same length.")

        N = len(labels)
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

        # Completing the loop
        data = data + data[:1]
        theta += theta[:1]

        fig, ax = plt.subplots(figsize=(figure_width, figure_height), subplot_kw={'polar': True})
        ax.fill(theta, data, color=color.value, alpha=opacity)
        ax.set_xticks(theta[:-1])  # Avoid repeating the first tick
        ax.set_xticklabels(labels)
        plt.title(chart_title)
        plt.show()