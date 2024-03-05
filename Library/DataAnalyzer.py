from typing import List, Union, Optional, Tuple

import pandas as pd

import warnings
warnings.filterwarnings("ignore")

class DataAnalyzer:
    def __init__(self):
        """
        Initializes the DataPreprocessor class.
        """
        pass    
    
    def calculateAverageOfNumericColumns(self, dataframe: pd.DataFrame) -> Optional[Union[float, pd.Series]]:
        """
        Calculates the average of all numeric columns in a pandas DataFrame.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame containing the numerical columns.

        Returns:
        - Optional[Union[float, pd.Series]]: A Series containing the average values of all numeric columns in the DataFrame.
          Returns None if there are no numeric columns.
        """
        # Filter numeric columns using select_dtypes
        numeric_columns = dataframe.select_dtypes(include=['int', 'float'])

        if not numeric_columns.empty:
            # Calculate the mean for each numeric column
            column_means = numeric_columns.mean()
            return column_means
        else:
            return None
           
    def findColumnsByTopValuePercentage(self, dataframe: pd.DataFrame, percentage: float) -> List[Tuple[str, float]]:
        """
        Identifies columns where the most frequent value exceeds a specified percentage threshold and returns a list of tuples.
        Each tuple contains the column name and the percentage of the most frequent value.  

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to be analyzed.
        - percentage (float): The percentage threshold (as a whole number, e.g., 50 for 50%).   

        Returns:
        - List[Tuple[str, float]]: A list of tuples, where each tuple contains a column name and the percentage of the most frequent value.
        """
        cols_with_high_percentage = {}
        for column in dataframe.columns:
            value_counts_normalized = dataframe[column].value_counts(normalize=True)
            max_percentage_value = value_counts_normalized.max() * 100

            if max_percentage_value > percentage:
                cols_with_high_percentage[column] = max_percentage_value
        return cols_with_high_percentage
    
    def findColumnsWithMissingValuesAboveThreshold(self, dataframe: pd.DataFrame, percentage: float) -> List[Tuple[str, float]]:
        """
        Identifies columns in a DataFrame where the percentage of missing values exceeds a specified threshold.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to be analyzed.
        - percentage (float): The threshold percentage of missing values (e.g., 20 for 20%).

        Returns:
        - List[Tuple[str, float]]: A list of tuples, where each tuple contains the column name and the percentage of missing values in that column, for columns exceeding the specified threshold.
        """
        rowsCount = dataframe.shape[0]
        cols_with_high_missing_values = []
        for column in dataframe.columns:
            missing_percentage = (dataframe[column].isnull().sum() / rowsCount) * 100
            if missing_percentage > percentage:
                cols_with_high_missing_values.append((column, missing_percentage))

        return cols_with_high_missing_values  
    
    def calculateNullValuesSum(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Calculates the sum of null values in each column of the DataFrame.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to analyze for null values.

        Returns:
        - pd.Series: A Series object representing the sum of null values in each column.
        """
        return dataframe.isnull().sum()
    
    def sortCorrelationsWithColumn(self, dataframe: pd.DataFrame, column: str) -> pd.Series:
        """
        Sorts correlations of a specified column with all other columns in the DataFrame in descending order.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to analyze.
        - column (str): The name of the column to sort correlations for.

        Returns:
        - pd.Series: A Series object containing the sorted absolute correlation values with the specified column.
        """
        if column not in dataframe.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")
        return abs(dataframe.corr()[column]).sort_values(ascending=False)
    
    def filterRowsByColumnRange(self, dataframe: pd.DataFrame, column: str, minValue: Union[int, float], maxValue: Union[int, float]) -> Optional[pd.DataFrame]:
        """
        Filters rows in a DataFrame based on a specified column's value range.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to filter.
        - column (str): The name of the column to apply the range filter.
        - minValue (Union[int, float]): The minimum value of the range.
        - maxValue (Union[int, float]): The maximum value of the range.

        Returns:
        - Optional[pd.DataFrame]: A DataFrame containing rows where the specified column's values are within the given range. Returns None if the column's data type is not suitable.
        """
        if dataframe[column].dtype in [float, int]:
            filtered_df = dataframe[(dataframe[column] >= minValue) & (dataframe[column] <= maxValue)]
            return filtered_df
        else:
            print("Source type is not suitable for this function. Suitable types: int, float.")
            return None
        
    def sortDataframe(self, dataframe: pd.DataFrame, column: Union[str, list], ascending: bool = True, permanent: bool = False) -> Union[None, pd.DataFrame]:
        """
        Sorts the given DataFrame according to the specified column(s) and order. Can optionally modify the DataFrame in place.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to be sorted.
        - column (Union[str, list]): Column name or list of column names to sort by.
        - ascending (bool): Determines if the sort should be in ascending order. Default is True.
        - permanent (bool): If True, the original DataFrame is modified in place. Default is False.

        Returns:
        - Union[None, pd.DataFrame]: None if permanent is True (DataFrame is modified in place), otherwise a new sorted DataFrame.

        Example to sort DataFrame by multiple columns:
            sorted_df = sortDataframe(df, column=['Age', 'Salary'], ascending=[False, True], permanent=False)
        """
        if permanent:
            dataframe.sort_values(by=column, ascending=ascending, inplace=True)
            return None
        else:
            return dataframe.sort_values(by=column, ascending=ascending, inplace=False)