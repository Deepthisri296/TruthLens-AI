def generate_explanation(predicted_class, confidence):

    predicted_class = str(
        predicted_class
    ).upper()

    confidence = float(
        confidence
    )

    if predicted_class == "FAKE":

        return (
            f"The model classified the image as FAKE "
            f"with {confidence:.2f}% confidence. "
            f"Grad-CAM highlights the regions that "
            f"contributed most strongly to the FAKE "
            f"prediction. These highlighted regions "
            f"represent visual features that had the "
            f"greatest influence on the model's decision."
        )

    return (
        f"The model classified the image as REAL "
        f"with {confidence:.2f}% confidence. "
        f"Grad-CAM highlights the regions that "
        f"contributed most strongly to the REAL "
        f"prediction. These highlighted regions "
        f"represent visual features that had the "
        f"greatest influence on the model's decision."
    )