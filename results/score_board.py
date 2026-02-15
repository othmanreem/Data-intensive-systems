import matplotlib.pyplot as plt


def create_score_board(weeks=None, scores=None, save_path=None, show=False):
    """
    Create a score board plot for R^2 scores over weeks.

    Parameters
    ----------
    weeks : list, optional
        List of week numbers. Default is [0, 1, 2].
    scores : list, optional
        List of R^2 scores. Default is [0, 0.51, 0.59].
    save_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    show : bool, optional
        Whether to display the plot. Default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.
    """
    if weeks is None:
        weeks = [0, 1, 2]
    if scores is None:
        scores = [0, 0.51, 0.59]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the data
    ax.plot(
        weeks,
        scores,
        marker='o',
        linestyle='-',
        color='b',
        linewidth=2,
        markersize=8
    )

    # Set labels and title
    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('R^2 Score', fontsize=12)
    ax.set_title(
        'R^2 Score Progression Over Weeks',
        fontsize=14,
        fontweight='bold'
    )

    # Set x-axis ticks to be exactly the weeks
    ax.set_xticks(weeks)
    ax.set_xticklabels([f'Week {w}' for w in weeks])

    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    # Annotate each point with its score
    for i, (x, y) in enumerate(zip(weeks, scores)):
        ax.annotate(
            f'{y:.2f}',
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )

    # Adjust layout to prevent clipping
    fig.tight_layout()

    # Save the plot if save_path is provided
    if save_path is not None:
        # we save it into the main directory
        fig.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")

    # Show the plot if requested
    if show:
        plt.show()

    return fig


if __name__ == "__main__":
    # Example usage with default data
    fig = create_score_board(save_path='score_board.png', show=True)
