import matplotlib.pyplot as plt


def create_score_board(save_path=None, show=False):
    """
    Create a score board plot for R^2 and F1 scores over assignments.

    Parameters
    ----------
    save_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    show : bool, optional
        Whether to display the plot. Default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.
    """
    # Assignment progression
    assignments = ['Start', 'A2', 'A3', 'A4', 'A5', 'A5b/A6']
    
    # R² Score progression (Regression)
    # Start=0, A2 baseline=0.52, A2 outliers=0.59, A4 RF=0.65, A5 Ensemble=0.7204, A5b retained
    r2_scores = [0, 0.52, 0.59, 0.65, 0.7204, 0.7204]
    
    # F1 Score progression (Classification)
    # Start=0, A2=N/A(0), A3 LDA=0.57, A4 RF=0.6110, A5 Ensemble=0.6484, A6 retained
    f1_scores = [0, 0, 0.57, 0.6110, 0.6484, 0.6484]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x_pos = range(len(assignments))

    # Plot R² scores
    ax1.plot(
        x_pos,
        r2_scores,
        marker='o',
        linestyle='-',
        color='#4361ee',
        linewidth=2,
        markersize=10
    )
    ax1.set_xlabel('Assignment', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title(
        'Regression: R² Score Progression',
        fontsize=14,
        fontweight='bold'
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(assignments)
    ax1.set_ylim(0, 0.85)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.axhline(y=0.7204, color='green', linestyle=':', alpha=0.5, label='Champion')
    
    for i, (x, y) in enumerate(zip(x_pos, r2_scores)):
        ax1.annotate(
            f'{y:.2f}' if y > 0 else '',
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontweight='bold'
        )

    # Plot F1 scores
    ax2.plot(
        x_pos,
        f1_scores,
        marker='s',
        linestyle='-',
        color='#06d6a0',
        linewidth=2,
        markersize=10
    )
    ax2.set_xlabel('Assignment', fontsize=12)
    ax2.set_ylabel('F1 Score (weighted)', fontsize=12)
    ax2.set_title(
        'Classification: F1 Score Progression',
        fontsize=14,
        fontweight='bold'
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(assignments)
    ax2.set_ylim(0, 0.85)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.axhline(y=0.6484, color='green', linestyle=':', alpha=0.5, label='Champion')
    
    for i, (x, y) in enumerate(zip(x_pos, f1_scores)):
        ax2.annotate(
            f'{y:.2f}' if y > 0 else '',
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontweight='bold'
        )

    # Adjust layout
    fig.suptitle('Model Performance Progression', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()

    # Save the plot if save_path is provided
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    # Show the plot if requested
    if show:
        plt.show()

    return fig


if __name__ == "__main__":
    # Generate the score board
    fig = create_score_board(save_path='score_board.png', show=True)
