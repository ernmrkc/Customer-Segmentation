from functools import reduce
from typing import List, Union, Optional, Any, Type
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import pandas as pd
import numpy as np
import re

import warnings
warnings.filterwarnings("ignore")

class DataPreprocessor:
    def __init__(self):
        """
        Initializes the DataPreprocessor class.
        """
        pass

    def mergeMultipleDataFrame(self, dataframes: List[pd.DataFrame], on_column: str = '', how: str = 'outer') -> pd.DataFrame:
        """
        Merges multiple DataFrames based on a specified column.

        Parameters:
        - dataframes (List[pd.DataFrame]): List of DataFrames to be merged.
        - on_column (str): The name of the column to merge on. This parameter cannot be an empty string.
        - how (str): Type of merge to be performed. Can be 'left', 'right', 'outer', 'inner'. Default is 'outer'.

        Returns:
        - df_merged (pd.DataFrame): The merged DataFrame.
        """
        if not on_column:
            raise ValueError("The 'on_column' parameter cannot be left empty.")

        df_merged = reduce(lambda left, right: pd.merge(left, right, on=[on_column], how=how), dataframes)
        return df_merged

    def addColumnWithValue(self, dataframe: pd.DataFrame, newColumnName: str, value: Any) -> pd.DataFrame:
        """
        Adds a new column to the specified DataFrame and assigns the given value to all rows in the new column.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to which the new column will be added.
        - newColumnName (str): The name of the new column to be added.
        - value (Any): The value to be assigned to all rows in the new column.

        Returns:
        - pd.DataFrame: The DataFrame with the new column added.
        """
        df_copy = dataframe.copy()
        df_copy[newColumnName] = value
        return df_copy
    
    def addDerivedDateColumn(self, dataframe: pd.DataFrame, newColumnName: str, byColumn: str, dateType: str) -> pd.DataFrame:
        """
        Adds a new column to the DataFrame derived from a specified date column. The new column contains date-related information such as day, month, year, month name, or weekday name.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to modify.
        - newColumnName (str): The name of the new column to add.
        - byColumn (str): The name of the date column from which to derive the new column.
        - dateType (str): Type of date information to add ('day', 'month', 'year', 'monthName', 'weekdayName').

        Returns:
        - pd.DataFrame: The DataFrame with the new column added.
        """
        df_copy = dataframe.copy()
        if df_copy[byColumn].isnull().sum() > 0:
            raise ValueError("The specified column contains empty rows. Please fill in the blank lines first.")

        if dateType == "day":
            df_copy[newColumnName] = df_copy[byColumn].dt.day
        elif dateType == "month":
            df_copy[newColumnName] = df_copy[byColumn].dt.month
        elif dateType == "year":
            df_copy[newColumnName] = df_copy[byColumn].dt.year
        elif dateType == "monthName":
            df_copy[newColumnName] = df_copy[byColumn].dt.month_name()
        elif dateType == "weekdayName":
            df_copy[newColumnName] = df_copy[byColumn].dt.day_name()
        else:
            raise ValueError("'dateType' is not suitable for this function. Suitable types: day, month, year, monthName, weekdayName. Please check 'dateType' parameter.")

        return df_copy
    
    def convertColumnToDateTime(self, dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Converts a specified column in the DataFrame to datetime format. 
        Checks if the column is of object type and if there are no null values before converting.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame containing the column to be converted.
        - column (str): The name of the column to convert to datetime format.

        Returns:
        - Optional[str]: The data type of the column after the conversion attempt, or None if conversion was not attempted due to errors.
        """
        df_copy = dataframe.copy()
        if df_copy[column].dtype == object or df_copy[column].dtype == 'datetime64[ns]':
            try:
                df_copy[column] = pd.to_datetime(df_copy[column], errors='coerce')
                return df_copy[column].dtype
            except Exception as e:
                print(f"Error converting column to datetime: {e}")
        else:
            print("Column data type is not suitable for this function. Suitable types: object or datetime.")

        return df_copy
    
    def updateColumnValuesBasedOnCondition(self, dataframe: pd.DataFrame, column: str, condition: str, conditionValue: Optional[Any] = None, isTrue: Any = None, isFalse: Any = None) -> pd.DataFrame:
        """
        Updates the values in a DataFrame column based on a specified condition and returns the modified DataFrame.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to modify.
        - column (str): The name of the column to update.
        - condition (str): The condition to evaluate ('isnull', 'notnull', 'lessThan', 'moreThan', 'equals').
        - conditionValue (Optional[Any]): The value to compare against when the condition is 'lessThan', 'moreThan', or 'equals'. Default is None.
        - isTrue (Any): The value to assign where the condition is True.
        - isFalse (Any): The value to assign where the condition is False.

        Returns:
        - pd.DataFrame: A modified copy of the DataFrame with updated values based on the specified condition.
        """
        # Create a copy of the DataFrame to avoid modifying the original
        df_copy = dataframe.copy()

        if condition == "isnull":
            df_copy[column] = np.where(df_copy[column].isnull(), isTrue, isFalse)
        elif condition == "notnull":
            df_copy[column] = np.where(df_copy[column].notnull(), isTrue, isFalse)
        elif condition == "lessThan":
            if conditionValue is not None:
                df_copy[column] = np.where(df_copy[column] < conditionValue, isTrue, isFalse)
            else:
                raise ValueError("conditionValue must be provided for 'lessThan' condition.")
        elif condition == "moreThan":
            if conditionValue is not None:
                df_copy[column] = np.where(df_copy[column] > conditionValue, isTrue, isFalse)
            else:
                raise ValueError("conditionValue must be provided for 'moreThan' condition.")
        elif condition == "equals":
            if conditionValue is not None:
                df_copy[column] = np.where(df_copy[column] == conditionValue, isTrue, isFalse)
            else:
                raise ValueError("conditionValue must be provided for 'equals' condition.")
        else:
            raise ValueError("Invalid condition provided. Please check 'condition' parameter.")

        return df_copy
        
    def dropColumnsFromDataFrame(self, dataframe: pd.DataFrame, columns: Union[str, List[str]]) -> pd.DataFrame:
        """
        Drops specified columns from a DataFrame.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to modify.
        - columns (Union[str, List[str]]): A single column name or a list of column names to be removed from the DataFrame.

        Returns:
        - pd.DataFrame: The DataFrame with the specified columns removed.
        """
        df_copy = dataframe.copy()
        if isinstance(columns, str):
            columns = [columns]
        columns = [column.strip() for column in columns]  # Trim whitespace
        df_copy = df_copy.drop(columns=columns, axis=1)
        return df_copy
    
    def replaceCharactersInColumn(self, dataframe: pd.DataFrame, column: str, oldChar: str, newChar: str) -> pd.DataFrame:
        """
        Replaces all occurrences of a specified character(s) with another character(s) in a given column of a DataFrame copy,
        without modifying the original DataFrame, and returns the modified DataFrame copy.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame containing the column to modify.
        - column (str): The name of the column in which characters are to be replaced.
        - oldChar (str): The character(s) to be replaced.
        - newChar (str): The character(s) to replace with.

        Returns:
        - pd.DataFrame: A copy of the DataFrame with the specified column's characters replaced, without altering the original DataFrame.
        """
        # Create a copy of the DataFrame to ensure the original is not modified
        df_copy = dataframe.copy()
        # Ensure the column exists in the DataFrame copy
        if column not in df_copy.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")

        # Replace the characters in the copy
        df_copy[column] = df_copy[column].astype(str).str.replace(oldChar, newChar, regex=False)

        return df_copy
    
    def cleanTextInColumn(self, dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Cleans the text in a specified column of a DataFrame by removing non-word characters,
        underscores, specific symbols, and digits. It also removes newline and carriage return characters.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame containing the column to clean.
        - column (str): The name of the column in which text cleaning is to be performed.

        Returns:
        - pd.Series: The cleaned text column as a pandas Series.
        """
        # Ensure the column exists in the DataFrame
        df_copy = dataframe.copy()
        if column not in df_copy.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")

        # Replace non-word characters (excluding spaces), digits, and specific symbols with an empty string
        df_copy[column] = df_copy[column].astype(str).str.replace("[^\w\s]", "", regex=True)
        df_copy[column] = df_copy[column].replace("_", "", regex=True)
        df_copy[column] = df_copy[column].replace("½", "", regex=False) # No need for regex, it's a direct character
        df_copy[column] = df_copy[column].replace("\d+", "", regex=True)
        df_copy[column] = df_copy[column].replace("\n", "", regex=False) # Direct character, regex not required
        df_copy[column] = df_copy[column].replace("\r", "", regex=False) # Direct character, regex not required

        return df_copy

    def removeOutliersByQuantile(self, dataframe: pd.DataFrame, column: str, quantileMin: float, quantileMax: float) -> pd.DataFrame:
        """
        Removes outliers from a specified column in a DataFrame based on quantile thresholds.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to modify.
        - column (str): The name of the column from which outliers are to be removed.
        - quantileMin (float): The lower quantile threshold. Values below this quantile will be considered outliers.
        - quantileMax (float): The upper quantile threshold. Values above this quantile will be considered outliers.

        Returns:
        - pd.DataFrame: A DataFrame with outliers removed from the specified column.
        """
        if column not in dataframe.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")

        q_Max = dataframe[column].quantile(quantileMax)
        q_Min = dataframe[column].quantile(quantileMin)
        filtered_df = dataframe[(dataframe[column] <= q_Max) & (dataframe[column] >= q_Min)]

        return filtered_df
    
    def changeColumnDataType(self, dataframe: pd.DataFrame, column: str, target_type: Type) -> pd.DataFrame:
        """
        Changes the data type of a specified column in a DataFrame copy, without modifying the original DataFrame.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame containing the column to modify.
        - column (str): The name of the column for which the data type is to be changed.
        - target_type (Type): The target data type (e.g., int, float, object).

        Returns:
        - pd.DataFrame: A copy of the DataFrame with the modified column data type.
        """
        # Create a copy of the DataFrame to ensure the original is not modified
        df_copy = dataframe.copy()

        if df_copy[column].isnull().sum() > 0:
            print("This column has empty rows. Please fill in the blank lines before calling this function.")
            return df_copy

        try:
            df_copy[column] = df_copy[column].astype(target_type)
        except Exception as e:
            print(f"Error converting column '{column}' to {target_type}: {e}")
            return df_copy

        return df_copy
    
    def fillMissingAndConvertType(self, dataframe, column, target_type, fill_strategy):
        """
        Fills missing values and converts the data type of a specified column in a DataFrame copy.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame containing the column to be modified.
        - column (str): The name of the column for which the data type will be changed.
        - target_type (type): The target data type to convert the column into (e.g., int, float).
        - fill_strategy (str): Strategy to fill missing values ('min', 'max', 'mean', 'median', 'mode').

        Returns:
        - pd.DataFrame: A copy of the DataFrame with the specified column's data type converted and missing values filled.
        """
        # Create a copy of the DataFrame to ensure the original is not modified
        df_copy = dataframe.copy()
        # Convert non-numeric values to NaN
        df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')

        # Fill missing values based on the specified strategy
        if fill_strategy == "max":
            fill_value = df_copy[column].max()
        elif fill_strategy == "min":
            fill_value = df_copy[column].min()
        elif fill_strategy == "mean":
            fill_value = df_copy[column].mean()
        elif fill_strategy == "median":
            fill_value = df_copy[column].median()
        elif fill_strategy == "mode":
            # mode() returns a Series, use the first value if it exists, or default to mean if empty
            fill_value = df_copy[column].mode()[0] if not df_copy[column].mode().empty else df_copy[column].mean()
        else:
            raise ValueError("Invalid fill strategy. Please choose from 'min', 'max', 'mean', 'median', 'mode'.")

        # Fill missing values with the determined fill value
        df_copy[column].fillna(fill_value, inplace=True)

        # Convert the column to the target data type
        df_copy[column] = df_copy[column].astype(target_type)

        return df_copy


    def extractAndConcatenateNumbers(self, dataframe: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Extracts numbers from the specified columns of a DataFrame copy and concatenates them into a single string,
        without modifying the original DataFrame.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to modify.
        - columns (List[str]): A list of column names from which numbers will be extracted and concatenated.

        Returns:
        - pd.DataFrame: A copy of the DataFrame with the specified columns updated to contain only the extracted numbers.
        """
        # Create a copy of the DataFrame to ensure the original is not modified
        df_copy = dataframe.copy()

        def find_and_concatenate_numbers(text: str) -> str:
            """Finds all numbers in a string and concatenates them."""
            numbers = re.findall(r'\d+', text)
            return ''.join(numbers)

        for column in columns:
            if column in df_copy.columns:
                df_copy[column] = df_copy[column].astype(str).apply(find_and_concatenate_numbers)
            else:
                raise ValueError(f"Column '{column}' not found in DataFrame.")

        return df_copy
    
    def renameOrTransformColumnNames(self, dataframe: pd.DataFrame, process_type: str = "none", old_name: Union[str, None] = None, new_name: Union[str, None] = None) -> pd.DataFrame:
        """
        Changes the name of a specific column or transforms all column names based on the process type.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame whose column(s) are to be renamed or transformed.
        - process_type (str): Process type to determine the operation ('none', 'lower', 'upper'). Default is 'none'.
        - old_name (Union[str, None]): The current name of the column to be renamed. Required if process_type is 'none'.
        - new_name (Union[str, None]): The new name for the column. Required if process_type is 'none'.

        Returns:
        - None: The function modifies the DataFrame in place and does not return anything.
        """
        df_copy = dataframe.copy()
        
        if process_type == "none":
            if old_name is not None and new_name is not None:
                df_copy.rename(columns={old_name: new_name}, inplace=True)
            else:
                raise ValueError("Both 'old_name' and 'new_name' must be provided when 'process_type' is 'none'.")
        elif process_type == "lower":
            df_copy.columns = df_copy.columns.str.lower()
        elif process_type == "upper":
            df_copy.columns = df_copy.columns.str.upper()
        else:
            raise ValueError(f"Invalid 'process_type' value: {process_type}. Choose from 'none', 'lower', 'upper'.")
        return df_copy
        
    def resetDataFrameIndex(self, dataframe: pd.DataFrame, drop: bool = False) -> pd.DataFrame:
        """
        Resets the index of the DataFrame, optionally dropping the old index.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame whose index is to be reset.
        - drop (bool): If True, the old index is dropped and not added as a column in the DataFrame. Default is False.

        Returns:
        - pd.DataFrame: A new DataFrame with the index reset.
        """
        df_copy = dataframe.copy()
        return df_copy.reset_index(drop=drop)
    
    def fillMissingWithRandomNumeric(self, dataframe: pd.DataFrame, column: str, min_value: Union[int, float] = 0, max_value: Union[int, float] = 100, data_type: Type = int) -> pd.DataFrame:
        """
        Fills missing (NaN) values in a specified column of a DataFrame with random numeric values,
        either integers or floats, based on the column's data type.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame containing the column to modify.
        - column (str): The name of the column where missing values will be filled with random numbers.
        - min_value (Union[int, float]): The minimum value for generating random numbers. Default is 0.
        - max_value (Union[int, float]): The maximum value for generating random numbers. Default is 100.

        Returns:
        - None: The function modifies the DataFrame in place.
        """
        df_copy = dataframe.copy()
        if column not in df_copy.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")
        
        is_int = data_type is int
        num_missing = df_copy[column].isnull().sum()
        
        if is_int:
            random_values = np.random.randint(min_value, max_value + 1, size=num_missing)
        else:
            random_values = np.random.uniform(min_value, max_value, size=num_missing)
        
        df_copy.loc[df_copy[column].isnull(), column] = random_values
        df_copy[column] = df_copy[column].astype(data_type)
        
        return df_copy
        
    def scaleColumnStandard(self, dataframe: pd.DataFrame, byColumn: str, newColumnName: str) -> pd.DataFrame:
        """
        Scales the data in the specified column using StandardScaler, adding the scaled data to a new column in the DataFrame.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame containing the column to be scaled.
        - byColumn (str): The name of the column whose data is to be scaled.
        - newColumnName (str): The name of the new column where the scaled data will be stored.

        Returns:
        - pd.DataFrame: The DataFrame with the new column containing the scaled data.

        Note:
        - StandardScaler transforms data to have a mean of 0 and a standard deviation of 1.
        - It is more resilient to outliers and often used as the default scaling method for many machine learning models.
        """
        df_copy = dataframe.copy()
        if byColumn not in df_copy.columns:
            raise ValueError(f"Column '{byColumn}' not found in DataFrame.")

        scaler = StandardScaler()
        # Reshape is used as fit_transform expects 2D array
        scaled_data = scaler.fit_transform(df_copy[[byColumn]].values.reshape(-1, 1))
        df_copy[newColumnName] = scaled_data.flatten()  # Convert back to 1D array to match DataFrame column format

        return df_copy

    def scaleColumnMinMax(self, dataframe: pd.DataFrame, byColumn: str, newColumnName: str, minRange: Union[int, float] = 0, maxRange: Union[int, float] = 1) -> pd.DataFrame:
        """
        Scales the data in the specified column using MinMaxScaler, adding the scaled data to a new column in the DataFrame.
        This scaling squeezes the data between minRange and maxRange.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame containing the column to be scaled.
        - byColumn (str): The name of the column whose data is to be scaled.
        - newColumnName (str): The name of the new column where the scaled data will be stored.
        - minRange (Union[int, float]): The minimum value of the scale. Default is 0.
        - maxRange (Union[int, float]): The maximum value of the scale. Default is 1.

        Returns:
        - pd.DataFrame: The DataFrame with the new column containing the scaled data.

        Note:
        - MinMax scaling may be sensitive to outliers, as it relies on the minimum and maximum values of the data.
        - It is often used when the model input requires data scaled between [0, 1], such as with neural network models.
        """
        df_copy = dataframe.copy()
        if byColumn not in df_copy.columns:
            raise ValueError(f"Column '{byColumn}' not found in DataFrame.")

        scaler = MinMaxScaler(feature_range=(minRange, maxRange))
        # fit_transform expects a 2D array, hence the double brackets around byColumn
        scaled_data = scaler.fit_transform(df_copy[[byColumn]].values.reshape(-1, 1))
        df_copy[newColumnName] = scaled_data.flatten()  # Convert back to 1D array to match DataFrame column format

        return df_copy