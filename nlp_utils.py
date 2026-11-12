
import re

positive_words = {
    "good", "great", "excellent", "amazing",
    "awesome", "love", "happy", "best",
    "friendly", "helpful", "thanks", "thank",
    "wonderful", "perfect", "smooth", "quick",
    "comfortable", "appreciate", "enjoy",
    "fantastic", "nice", "easy", "professional",
    "outstanding", "brilliant", "pleased",
    "delighted", "efficient", "polite", "recommend"
}

negative_words = {
    "bad", "worst", "terrible", "awful",
    "delay", "delayed", "cancelled", "cancel",
    "rude", "lost", "missing", "angry",
    "hate", "poor", "late", "problem",
    "issue", "unhappy", "disappointed",
    "frustrated", "horrible", "broken",
    "worse", "waiting", "stuck",
    "unhelpful", "disgusting", "annoying",
    "failure", "unacceptable"
}

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        "",
        text
    )

    text = re.sub(
        r"@\w+",
        "",
        text
    )

    text = re.sub(
        r"#(\w+)",
        r"\1",
        text
    )

    text = re.sub(
        r"\d+",
        "",
        text
    )

    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def heuristic_sentiment(text):

    tokens = clean_text(text).split()

    positive_score = 0
    negative_score = 0

    negations = {"not", "no", "never"}

    for i, word in enumerate(tokens):

        is_negated = any(
            previous in negations
            for previous in tokens[max(0, i-2):i]
        )

        if word in positive_words:

            if is_negated:
                negative_score += 1
            else:
                positive_score += 1

        elif word in negative_words:

            if is_negated:
                positive_score += 1
            else:
                negative_score += 1

    if positive_score > negative_score:
        prediction = "positive"

    elif negative_score > positive_score:
        prediction = "negative"

    else:
        prediction = "neutral"

    return prediction, positive_score, negative_score
