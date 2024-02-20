# pylint: disable=line-too-long
""" benchmark hybrid search """

import time
import json
import os
from dotenv import load_dotenv
import weaviate
import matplotlib.pyplot as plt
import numpy as np


# Load environment variables
load_dotenv()
schema_name = os.getenv("COLLNAME")
openai_key = os.getenv("API_KEY")
whost = os.getenv("WHOST")
wport = os.getenv("WPORT")
search_len = int(os.getenv("LIST_LEN"))  # how many results from the search
benchmark_size = int(os.getenv("BENCH_SIZE"))  # number of objects in the becnhmark file


# FUNCTIONS DEFINITIONS


def list_rank(stringlist, string):
    """arguments:
        list of strings
        one string
    returns:
        the 1 based item position for a match
        0 if the string is not found in the list
    """
    try:
        return stringlist.index(string) + 1
    except ValueError:
        return 0


def process_rank_frequencies(rank_frequency, total_counts):
    """Calculate and return rank frequencies and percentages."""
    percentages = {
        "False Negatives": (
            (rank_frequency.get(0, 0) / total_counts) * 100 if total_counts else 0
        ),
        "First Place": (
            (rank_frequency.get(1, 0) / total_counts) * 100 if total_counts else 0
        ),
        "First Five": sum(
            count for rank, count in rank_frequency.items() if 0 < rank <= 5
        )
        / total_counts
        * 100,
        "First Ten": sum(
            count for rank, count in rank_frequency.items() if 0 < rank <= 10
        )
        / total_counts
        * 100,
        f"First {search_len}": sum(
            count for rank, count in rank_frequency.items() if 0 < rank <= search_len
        )
        / total_counts
        * 100,
    }
    return percentages


def main():
    """Main function to process and plot Weaviate benchmark results."""
    alpha_values_list = [
        i / 10.0 for i in range(11)
    ]  # hybryd search alpha values to test
    benchfn = f"resources/italian-wines-{benchmark_size}-3-keywords-bench.json"  # Ensure this path is correct

    with open(benchfn, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Connect to Weaviate instance
    wclient = weaviate.connect_to_local(
        host=whost,
        port=wport,
        headers={
            "X-OpenAI-Api-Key": openai_key,  #  for generative queries
        },
    )

    if wclient.is_ready():
        print("Client is ready")
        articles = wclient.collections.get(schema_name)
        response = articles.aggregate.over_all(total_count=True)
        articles_number = response.total_count
        rank_counts = {
            alpha: {"zero": 0, "one": 0} for alpha in alpha_values_list
        }  # dict to collect stats for each alpha value
        execution_times = []
        for alpha in alpha_values_list:
            print(f"Benchmarking for alpha value of {alpha}")
            for wine in data:
                title = wine["title"]
                keywords = wine["keywords"]
                start_time = time.time()
                response = articles.query.hybrid(
                    query=" ".join(keywords),
                    limit=search_len,
                    alpha=alpha,  # alpha 1=pure vector, 0=pure keyword
                )
                end_time = time.time()  # End timing
                execution_time = end_time - start_time
                execution_times.append(execution_time)
                id_from_bench = title
                ids_from_search = [obj.properties["title"] for obj in response.objects]
                rank = list_rank(ids_from_search, id_from_bench)

                # save the rank in a number of "bins"
                if rank == 0:
                    rank_counts[alpha]["zero"] += 1
                else:
                    # Increment "found" for any rank value different from zero
                    if "found" not in rank_counts[alpha]:
                        rank_counts[alpha]["found"] = 1
                    else:
                        rank_counts[alpha]["found"] += 1

                    if rank == 1:
                        rank_counts[alpha]["one"] += 1

                # Increment "first3" for ranks 1, 2, or 3
                if rank in [1, 2, 3]:
                    if "first3" not in rank_counts[alpha]:
                        rank_counts[alpha]["first3"] = 1
                    else:
                        rank_counts[alpha]["first3"] += 1

                # Increment "first5" for ranks 1 through 5
                if rank in [1, 2, 3, 4, 5]:
                    if "first5" not in rank_counts[alpha]:
                        rank_counts[alpha]["first5"] = 1
                    else:
                        rank_counts[alpha]["first5"] += 1

                # Increment "first10" for ranks 1 through 10
                if rank in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                    if "first10" not in rank_counts[alpha]:
                        rank_counts[alpha]["first10"] = 1
                    else:
                        rank_counts[alpha]["first10"] += 1
        avgquerytime = round((sum(execution_times) / len(execution_times)) * 1000)
        print(
            f"Average search time over {len(execution_times)} runs is {avgquerytime} milliseconds"
        )
    wclient.close()

    # Plotting
    plt.figure(figsize=(10, 6))
    # Prepare the data for plotting
    alphas_sorted = sorted(alpha_values_list)  # Ensure alphas are sorted for plotting
    # Converting alphas_sorted to numerical indices for plotting
    alphas_indices = np.arange(len(alphas_sorted))

    # Data preparation
    zeros = [rank_counts[alpha].get("zero", 0) for alpha in alphas_sorted]
    found = [rank_counts[alpha].get("found", 0) for alpha in alphas_sorted]
    ones = [rank_counts[alpha].get("one", 0) for alpha in alphas_sorted]

    # Using a fixed bar width that should work well for a typical number of bars
    bar_width = 0.35

    # Stacked bar plots for "zero" and "found"
    plt.bar(alphas_indices, zeros, width=bar_width, label="Not found", color="grey")
    plt.bar(
        alphas_indices,
        found,
        bottom=zeros,
        width=bar_width,
        label="Found",
        color="lightblue",
    )

    # Line plot adjustments to ensure they align with the center of the bars
    # Adjusting the x values for line plots by adding half of the bar width
    # adjusted_alphas_indices = alphas_indices + bar_width / 2

    # Line plot for "ones"
    plt.plot(
        alphas_indices,
        ones,
        label="Top result",
        marker="x",
        linestyle="-",
        color="green",
    )

    # Plot for other rank_counts[alpha] elements with distinct colors
    additional_keys = set(rank_counts[alphas_sorted[0]].keys()) - {
        "zero",
        "one",
        "found",
    }
    colors = ["lightgreen", "lightcoral", "purple"]  # Assign colors for additional keys

    for key, color in zip(additional_keys, colors):
        counts = [rank_counts[alpha].get(key, 0) for alpha in alphas_sorted]
        plt.plot(
            alphas_indices,
            counts,
            label=key,
            marker=".",
            linestyle="-",
            color=color,
        )

    # Adding titles, labels, and legend
    plt.title("Benchmark of hybrid search for several alpha values")
    plt.xlabel("Alpha")
    plt.ylabel("Counts")
    plt.xticks(
        alphas_indices, alphas_sorted, rotation=45
    )  # Set x-ticks to show alpha values
    plt.legend(loc="upper left")
    plt.grid(True)

    # Adding the annotation below the Alpha label
    # Updated to use normalized axes coordinates properly
    plt.text(
        0.5,  # Centered on the x-axis
        -0.15,  # Position below the axis labels, adjust this as needed
        f"Avg. query execution time: {avgquerytime} ms. on {articles_number} objects",
        ha="center",  # Horizontally align the text to center
        va="top",  # Vertically align the text
        transform=plt.gca().transAxes,  # Use axes transform
    )
    plt.tight_layout()

    # Adding "sparse" and "dense" labels with adjusted positions
    lab_vertical_position = -0.1
    plt.text(
        0.0,
        lab_vertical_position,
        "sparse",
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
    )

    right_most_position = 1.0  # Assuming the last label is at the far right
    plt.text(
        right_most_position,
        lab_vertical_position,
        "dense",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
    )

    # Save the plot to a file
    plt.savefig("resources/hybrid_search_3-keywords_benchmark.png")
    # Display the plot
    plt.show()


if __name__ == "__main__":
    main()
