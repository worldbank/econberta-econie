import fitz
from os.path import join
import logging
import pandas as pd
import numpy as np
import ast


logging.basicConfig(level=logging.INFO)


def load_pdf(PATH_DATA: str, file: str):
    """
    Converts a PDF file into a list of images.

    :param PATH_DATA: Path to the directory containing the PDF file.
    :param file: Name of the PDF file without the extension.
    :return: document object.
    """

    # logging.info(f"Loading PDF from {file}...")

    # Open the PDF file
    doc = fitz.open(join(PATH_DATA, f"{file}.pdf"))

    return doc


def get_bboxes(doc, pdf):
    """
    Extract bounding boxes from a document and return as a DataFrame.

    :param doc: The document to extract bounding boxes from.
    :param pdf: The name of the PDF file.
    :return: A DataFrame of bounding boxes.
    """
    bboxes = []

    # Iterate over each page in the document
    for i, page in enumerate(doc):
        # Extract blocks from the page
        blocks = ast.literal_eval(page.get_text("json")).get("blocks")

        # Extract bounding boxes from each line in each block
        for block in blocks:
            for line in block.get("lines", []):
                # Append bounding box coordinates, text, and page number to the list
                bbox = (
                    [pdf]
                    + line.get("bbox")
                    + ["".join([span.get("text") for span in line.get("spans")])]
                    + [i]
                )
                bboxes.append(bbox + list(page.rect)[2:])

    # Convert the list of bounding boxes to a DataFrame
    df = pd.DataFrame(
        bboxes,
        columns=[
            "filename",
            "x1",
            "y1",
            "x2",
            "y2",
            "text",
            "#page",
            "width",
            "height",
        ],
    )

    # Round the bounding box coordinates to 2 decimal places
    for col in ["x1", "y1", "x2", "y2"]:
        df[col] = df[col].round(2)

    return df


def aggregate_lines(df):
    """
    Aggregate lines in a DataFrame based on the number of columns and their positions.

    :param df: The DataFrame to aggregate.
    :return: The aggregated DataFrame.
    """
    # Determine the number of columns and their positions
    n_columns, peaks_1, peaks_2 = get_num_columns(df)

    # Assign each line to a column based on its x1 or x2 value
    if n_columns == 1:
        df["column"] = 1
    elif len(peaks_1) >= len(peaks_2):
        df["column"] = df["x1"].apply(lambda x: find_closest_index_x1(peaks_1, x))
    else:
        df["column"] = df["x2"].apply(lambda x: find_closest_index_x2(peaks_2, x))

    # Sort the DataFrame by page number, column number, and x1 value
    df = df.sort_values(["#page", "column", "x1"])

    # Set y1, y2 as integers
    df["y1_int"] = df["y1"].astype(int)
    df["y2_int"] = df["y2"].astype(int)

    # Group lines by filename, page number, column number, y1, and y2
    # Aggregate x1, x2, text, width, and height values for each group
    df = (
        df.groupby(["filename", "#page", "column", "y1_int", "y2_int"], as_index=False)
        .agg(
            {
                "x1": "min",
                "x2": "max",
                "y1": "min",
                "y2": "max",
                "text": " ".join,
                "width": "first",
                "height": "first",
            }
        )
        .sort_values(["#page", "column", "y1", "x1"])
        .reset_index(drop=True)
    )

    df.loc[:, "group"] = (
        (df.loc[:, "y1"] <= df.loc[:, "y1"].shift())
        | (df.loc[:, "y2"] >= df.loc[:, "y2"].shift())
    ).cumsum()

    df = (
        df.groupby(["filename", "#page", "column", "group"], as_index=False)
        .agg(
            {
                "x1": "min",
                "y1": "min",
                "x2": "max",
                "y2": "max",
                "text": " ".join,
                "width": "first",
                "height": "first",
            }
        )
        .sort_values(["filename", "#page", "column", "x1", "y1"])
        .reset_index(drop=True)
    )

    # Add the number of columns to the DataFrame
    df["#column"] = n_columns

    # Sort the DataFrame by page number, column number, y1, and x1 value
    df = df.sort_values(["#page", "column", "y1", "x1"]).reset_index(drop=True)

    return df


def get_num_columns(df, n_bins=100):
    """
    Determine the number of columns in a dataframe based on the distribution of x1 and x2 values.

    :param df: A dataframe containing x1 and x2 values.
    :param n_bins: The number of bins to use for the histograms.
    :return: The number of columns, the x1 peaks, and the x2 peaks.
    """
    # Extract x1 and x2 values
    x1 = df.loc[:, "x1"]
    x2 = df.loc[:, "x2"]

    # Calculate histograms for x1 and x2
    hist_1 = np.histogram(x1, bins=n_bins)
    hist_2 = np.histogram(x2, bins=n_bins)

    # Calculate the 97.5th percentile for x1 and x2
    q975_x1 = np.quantile(hist_1[0], 0.5) + 3 * (
        np.quantile(hist_1[0], 0.95) - np.quantile(hist_1[0], 0.5)
    )
    q975_x2 = np.quantile(hist_2[0], 0.5) + 3 * (
        np.quantile(hist_2[0], 0.95) - np.quantile(hist_2[0], 0.5)
    )

    # Identify peaks in the histograms
    _peaks_1 = [hist_1[1][i] for i in np.where(hist_1[0] > q975_x1)[0]]
    _peaks_2 = [hist_2[1][i] for i in np.where(hist_2[0] > q975_x2)[0]]

    # Combine peaks that are closer than half the median of diff between x2 and x1
    threshold = np.median(x2 - x1) / 2
    peaks_1 = _peaks_1[:1]
    peaks_2 = _peaks_2[:1]
    for peak_1 in _peaks_1[1:]:
        if peak_1 - peaks_1[-1] > threshold:
            peaks_1.append(peak_1)
    for peak_2 in _peaks_2[1:]:
        if peak_2 - peaks_2[-1] > threshold:
            peaks_2.append(peak_2)

    # Determine the number of columns
    n_col = max(len(peaks_1), len(peaks_2), 1)

    return n_col, peaks_1, peaks_2


def find_closest_index_x1(lst, value):
    """
    Find the index of the closest value in a list that is less than or equal to a given value.

    :param lst: A list of values.
    :param value: The value to find the closest index for.
    :return: The index of the closest value in the list that is less than or equal to the given value.
    """
    for i in range(len(lst) - 1):
        if lst[i] <= value < lst[i + 1] - 30:
            return i + 1
    return len(lst)


def find_closest_index_x2(lst, value):
    """
    Find the index of the closest value in a list that is less than a given value.

    :param lst: A list of values.
    :param value: The value to find the closest index for.
    :return: The index of the closest value in the list that is less than the given value.
    """
    if value < lst[0]:
        return 1
    for i in range(1, len(lst)):
        if lst[i - 1] <= value < lst[i] - 30:
            return i + 1
    return len(lst)
