from PIL import Image
from app.data_processors import *
from app.models import *

def process_uploaded_image(
    upl_img
):
    """
    Process an uploaded image.

    :param upl_img:
        The uploaded image file.

    :return:
        The processed image.
    """
    img = Image.open(upl_img.file)
    img_processor = FbMlImageProcessor()
    processed_img = img_processor(img)
    return processed_img

def process_text(
    txt,
    max_length
):
    """
    Process some text.

    :param txt:
        The given text.

    :return:
        The processed text.
    """
    txt_processor = FbMlTextProcessor(max_length=max_length)
    processed_txt = txt_processor(txt)
    return processed_txt

def get_ppc_image(
    img
):
    """
    Given an uploaded image, returns the prediction, predicted probabilities, and predicted classes using the FbMlImageClassifier.
    
    :param upl_img:
        The uploaded image to be classified
        
    :return:
        tuple: (prediction, probas, classes)
    """
    img_classifier = FbMlImageClassifier()
    prediction = img_classifier.predict(img)
    probas = img_classifier.predict_proba(img)
    classes = img_classifier.predict_classes(img)

    return prediction.tolist(), probas.tolist(), classes

def get_ppc_text(
    txt
):
    """
    Given some text, returns the prediction, predicted probabilities, and predicted classes using the FbMlTextClassifier.
    
    :param txt:
        The text to be classified
        
    :return:
        tuple: (prediction, probas, classes)
    """
    txt_classifier = FbMlTextClassifier()
    prediction = txt_classifier.predict(txt)
    probas = txt_classifier.predict_proba(txt)
    classes = txt_classifier.predict_classes(txt)

    return prediction.tolist(), probas.tolist(), classes

def get_ppc_combined(
    txt,
    img
):
    """
    Given some text and an image, returns the prediction, predicted probabilities, and predicted classes using the FbMlCombinedClassifier.
    
    :param txt:
        The text to be classified    
    :param img
        The text to be classified
        
    :return:
        tuple: (prediction, probas, classes)
    """
    combined_classifier = FbMlCombinedClassifier()
    prediction = combined_classifier.predict(txt, img)
    probas = combined_classifier.predict_proba(txt, img)
    classes = combined_classifier.predict_classes(txt, img)

    return prediction.tolist(), probas.tolist(), classes


def classify_image(
    upl_img
):
    """
    Given an uploaded image, processes the image and then return the prediction, predicted probabilities, and predicted classes.
    
    :param upl_img:
        The uploaded image to be classified
        
    :return:
        tuple: (prediction, probas, classes)
    """
    processed_img = process_uploaded_image(upl_img)

    prediction, probas, classes = get_ppc_image(processed_img)

    return prediction, probas, classes


def classify_text(
    txt
):
    """
    Given some text, processes the text and then return the prediction, predicted probabilities, and predicted classes.
    
    :param txt:
        The text to be classified
        
    :return:
        tuple: (prediction, probas, classes)
    """
    processed_txt = process_text(txt, max_length=100)

    prediction, probas, classes = get_ppc_text(processed_txt)

    return prediction, probas, classes


def classify_combined(
    upl_img,
    txt
):
    """
    Given an image and some text, processes the image and text and then return the prediction, predicted probabilities, and predicted classes.
    
    :param upl_img:
        The uploaded image
    :param txt:
        The text
        
    :return:
        tuple: (prediction, probas, classes)
    """
    processed_img = process_uploaded_image(upl_img)
    processed_txt = process_text(txt, max_length=50)

    prediction, probas, classes = get_ppc_combined(processed_txt, processed_img)

    return prediction, probas, classes