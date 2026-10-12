# Based on seminar materials

# Don't forget to support cases when target_text == ''


def edit_distance(target_text: str | list[str], predicted_text: str | list[str]):
    dp = [
        [0 for j in range(len(predicted_text) + 1)] for i in range(len(target_text) + 1)
    ]

    for i in range(1, len(target_text) + 1):
        dp[i][0] = dp[i - 1][0] + 1

    for j in range(1, len(predicted_text) + 1):
        dp[0][j] = dp[0][j - 1] + 1

    for i in range(1, len(target_text) + 1):
        for j in range(1, len(predicted_text) + 1):
            dp[i][j] = min(
                dp[i][j - 1] + 1,  # D
                dp[i - 1][j] + 1,  # I
                dp[i - 1][j - 1]
                + int(target_text[i - 1] != predicted_text[j - 1]),  # C / S
            )

    return dp[-1][-1]


def calc_cer(target_text, predicted_text) -> float:
    """
    Calculates the CER between target and predicted texts.
    Returns the length of the predicted text in case of the empty target.
    Args:
        target_text(str): target text
        predicted_text(str): predicted text
    Returns:
        float: CER metric
    """
    if target_text == "":
        return len(predicted_text)
    return edit_distance(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    """
    Calculates the WER between target and predicted texts.
    Returns the length of the predicted text in case of the empty target.
    Args:
        target_text(str): target text
        predicted_text(str): predicted text
    Returns:
        float: WER metric
    """
    if len(target_text.split()) == 0:
        return len(predicted_text.split())
    return edit_distance(target_text.split(), predicted_text.split()) / len(
        target_text.split()
    )
