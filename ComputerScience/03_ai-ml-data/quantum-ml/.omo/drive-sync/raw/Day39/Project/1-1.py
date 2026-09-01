"""
01_dataset.py

Netflix Quantum ML Mini Project
--------------------------------
Step 1. Netflix Dataset Load
Step 2. Dataset Structure 확인
Step 3. Data Quality 분석
Step 4. Target 분포 분석
Step 5. Data Leakage 후보 확인

"""

from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

DATA_PATH = Path("netflix_titles.csv")


# ============================================================
# Utility
# ============================================================

def print_section(title: str) -> None:
    """출력 결과를 단계별로 구분하기 위한 함수"""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# STEP 1. Dataset Load
# ============================================================

def load_dataset(data_path: Path) -> pd.DataFrame:
    """
    Netflix Dataset을 CSV 파일에서 불러온다.

    Parameters
    ----------
    data_path : Path
        Netflix CSV 파일 경로

    Returns
    -------
    pd.DataFrame
        Netflix Dataset
    """

    print_section("STEP 1. Netflix Dataset Load")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset을 찾을 수 없습니다: {data_path.resolve()}"
        )

    df = pd.read_csv(data_path)

    print(f"Dataset Path : {data_path}")
    print(f"Rows         : {df.shape[0]:,}")
    print(f"Columns      : {df.shape[1]}")

    return df


# ============================================================
# STEP 2. Dataset Preview
# ============================================================

def show_dataset_preview(df: pd.DataFrame) -> None:
    """Dataset의 기본 내용을 확인한다."""

    print_section("STEP 2. Dataset Preview")

    print(df.head())


# ============================================================
# STEP 3. Dataset Structure
# ============================================================

def analyze_structure(df: pd.DataFrame) -> None:
    """Dataset의 컬럼과 데이터 타입을 확인한다."""

    print_section("STEP 3. Dataset Structure")

    print("[Columns]")

    for index, column in enumerate(df.columns, start=1):
        print(f"{index:2d}. {column}")

    print()
    print("[Data Types]")

    print(df.dtypes)

    print()
    print("[Dataset Info]")

    df.info()


# ============================================================
# STEP 4. Missing Values
# ============================================================

def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    컬럼별 결측치 개수와 비율을 분석한다.
    """

    print_section("STEP 4. Missing Value Analysis")

    missing_count = df.isnull().sum()

    missing_ratio = (
        df.isnull().sum()
        / len(df)
        * 100
    )

    missing_df = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_ratio(%)": missing_ratio
        }
    )

    missing_df = missing_df.sort_values(
        by="missing_count",
        ascending=False
    )

    print(
        missing_df.to_string(
            formatters={
                "missing_ratio(%)": "{:.2f}".format
            }
        )
    )

    return missing_df


# ============================================================
# STEP 5. Duplicate Analysis
# ============================================================

def analyze_duplicates(df: pd.DataFrame) -> None:
    """중복 데이터를 확인한다."""

    print_section("STEP 5. Duplicate Analysis")

    duplicate_count = df.duplicated().sum()

    print(f"Total Rows     : {len(df):,}")
    print(f"Duplicate Rows : {duplicate_count:,}")

    if duplicate_count == 0:
        print("Result         : 중복 데이터 없음")
    else:
        print("Result         : 중복 데이터 확인 필요")


# ============================================================
# STEP 6. Target Analysis
# ============================================================

def analyze_target(df: pd.DataFrame) -> None:
    """
    Classification Target인 type의 분포를 분석한다.
    """

    print_section("STEP 6. Target Analysis")

    if "type" not in df.columns:
        raise KeyError("'type' column이 존재하지 않습니다.")

    target_count = df["type"].value_counts()

    target_ratio = (
        df["type"]
        .value_counts(normalize=True)
        .mul(100)
    )

    target_summary = pd.DataFrame(
        {
            "count": target_count,
            "ratio(%)": target_ratio
        }
    )

    print(target_summary.round(2))

    print()
    print("Classification Problem")
    print("----------------------")
    print("Target : type")
    print("Class 0 candidate : Movie")
    print("Class 1 candidate : TV Show")


# ============================================================
# STEP 7. Duration Analysis
# ============================================================

def analyze_duration(df: pd.DataFrame) -> None:
    """
    duration 컬럼과 type의 관계를 확인한다.

    Movie는 '90 min', TV Show는 '1 Season'처럼 표현되므로
    duration이 Target 정보를 지나치게 직접 포함하는지 확인한다.
    """

    print_section("STEP 7. Duration / Data Leakage Analysis")

    required_columns = {"type", "duration"}

    if not required_columns.issubset(df.columns):
        print("type 또는 duration 컬럼이 없습니다.")
        return

    duration_df = (
        df[["type", "duration"]]
        .dropna()
    )

    print("[Movie Duration Samples]")

    print(
        duration_df[
            duration_df["type"] == "Movie"
        ].head(10).to_string(index=False)
    )

    print()
    print("[TV Show Duration Samples]")

    print(
        duration_df[
            duration_df["type"] == "TV Show"
        ].head(10).to_string(index=False)
    )

    print()
    print("[Leakage Check]")

    movie_min_ratio = (
        duration_df.loc[
            duration_df["type"] == "Movie",
            "duration"
        ]
        .str.contains("min", case=False, na=False)
        .mean()
        * 100
    )

    tv_season_ratio = (
        duration_df.loc[
            duration_df["type"] == "TV Show",
            "duration"
        ]
        .str.contains("Season", case=False, na=False)
        .mean()
        * 100
    )

    print(
        f"Movie 중 'min' 포함 비율       : "
        f"{movie_min_ratio:.2f}%"
    )

    print(
        f"TV Show 중 'Season' 포함 비율 : "
        f"{tv_season_ratio:.2f}%"
    )

    print()
    print(
        "주의: duration의 단위 자체가 Movie / TV Show 정보를 "
        "거의 직접적으로 나타낼 수 있습니다."
    )

    print(
        "따라서 이후 기본 Classification Feature에서는 "
        "duration을 제외합니다."
    )


# ============================================================
# STEP 8. Feature Candidate Review
# ============================================================

def review_feature_candidates(df: pd.DataFrame) -> None:
    """
    이후 Feature Engineering에 사용할 후보 컬럼을 확인한다.
    """

    print_section("STEP 8. Feature Candidate Review")

    candidates = [
        "release_year",
        "rating",
        "listed_in",
        "country",
        "description"
    ]

    print("다음 단계에서 사용할 Feature 후보")
    print()

    for column in candidates:

        if column not in df.columns:
            continue

        dtype = df[column].dtype
        missing = df[column].isnull().sum()

        print(
            f"{column:<15} "
            f"dtype={str(dtype):<10} "
            f"missing={missing:,}"
        )

    print()
    print("Feature Engineering Plan")
    print("------------------------")
    print("release_year -> Numeric Feature")
    print("rating       -> rating_encoded")
    print("listed_in    -> genre_count")
    print("country      -> country_count")
    print("description  -> description_length")


# ============================================================
# STEP 9. Dataset Summary
# ============================================================

def show_summary(df: pd.DataFrame) -> None:
    """01_dataset.py 분석 결과를 요약한다."""

    print_section("STEP 9. Dataset Analysis Summary")

    print(f"Dataset Size : {df.shape[0]:,} rows")
    print(f"Columns      : {df.shape[1]}")
    print("Target       : type")
    print("Classes      : Movie / TV Show")

    print()
    print("Important Finding")
    print("-----------------")

    print(
        "duration은 Target과 매우 강하게 연결되어 있으므로 "
        "Data Leakage 관점에서 주의가 필요합니다."
    )

    print()
    print("Next Step")
    print("---------")

    print(
        "02_preprocessing.py에서 Target Encoding과 "
        "Feature Engineering을 수행합니다."
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print_section(
        "Netflix Quantum ML Mini Project - Dataset Analysis"
    )

    # STEP 1
    df = load_dataset(DATA_PATH)

    # STEP 2
    show_dataset_preview(df)

    # STEP 3
    analyze_structure(df)

    # STEP 4
    analyze_missing_values(df)

    # STEP 5
    analyze_duplicates(df)

    # STEP 6
    analyze_target(df)

    # STEP 7
    analyze_duration(df)

    # STEP 8
    review_feature_candidates(df)

    # STEP 9
    show_summary(df)


if __name__ == "__main__":
    main()