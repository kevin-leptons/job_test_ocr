import locale

locale.setlocale(locale.LC_ALL, 'C')

import logging
import os
import sys
import traceback

import click
import cv2
import enchant
import filetype
import imutils
import numpy
import pdf2image
import PIL
import tesserocr

logger = logging.getLogger(__name__)


@click.command()
@click.option('--input', 'infile', required=True, help='path to input file')
@click.option('--output', 'outfile', required=True, help='path to output file')
@click.option('--verbose', is_flag=True, help='show what is going on')
def command(infile, outfile, verbose):
    try:
        setup_logging(verbose)
        extract_text(infile, outfile)
    except:
        logger.error(traceback.format_exc())
        sys.exit(1)


def setup_logging(enabled):
    if enabled is True:
        logging.basicConfig(level=logging.NOTSET)
    else:
        logging.disable()


def extract_text(infile, outfile):
    kind = filetype.guess(infile)

    if (kind.mime == 'image/jpeg') or (kind.mime == 'image/png'):
        extract_image_text(infile, outfile)
    elif kind.mime == 'application/pdf':
        extract_pdf_text(infile, outfile)
    else:
        raise Exception('input type is not supported')


def extract_image_text(infile, outfile):
    image = PIL.Image.open(infile)
    process_images([image], outfile)


def extract_pdf_text(infile, outfile):
    images = pdf2image.convert_from_path(infile)
    process_images(images, outfile)


def process_images(images, outfile):
    with open(outfile, 'w', encoding='utf-8') as f:
        with tesserocr.PyTessBaseAPI(lang='eng') as tess:
            for image in images:
                process_image(tess, image, f)


def process_image(tess, image, outfile):
    p_image = preprocess_image(image)
    text = recognize_text(tess, p_image)
    text = postprocess_text(text)
    write_output(text, outfile)


def recognize_text(tess, image):
    tess.SetImage(image)
    return tess.GetUTF8Text()


def preprocess_image(image):
    t = pil_to_cv_image(image)
    t = graycale_image(t)
    t = downsize_image(t)
    t = remove_shadow_image(t)
    t = denoise_image(t)
    t = binarization_image(t)
    t = deskew_image(t)

    return cv_to_pil_image(t)


def downsize_image(image):
    max_width = 1200
    _, w = image.shape[:2]

    if w <= max_width:
        return image

    return imutils.resize(image, width=max_width)


# from: https://stackoverflow.com/questions/44047819/increase-image-brightness-without-overflow/44054699#44054699
def remove_shadow_image(image):
    d = cv2.dilate(image, numpy.ones((7, 7), numpy.uint8))
    b = cv2.medianBlur(d, 21)
    f = 255 - cv2.absdiff(image, b)
    n = f.copy()
    cv2.normalize(f,
                  n,
                  alpha=0,
                  beta=255,
                  norm_type=cv2.NORM_MINMAX,
                  dtype=cv2.CV_8UC1)
    return n


def binarization_image(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def denoise_image(image):
    return cv2.fastNlMeansDenoising(image,
                                    h=10,
                                    templateWindowSize=7,
                                    searchWindowSize=21)


# from: https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
def deskew_image(image):
    coords = numpy.column_stack(numpy.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image,
                          m, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def postprocess_text(text):
    out = ''
    word = ''

    for symbol in text:
        # build temporary word.
        if is_english_symbol(symbol):
            word += symbol
            continue

        # symbol does not divide any word, push it to output.
        if len(word) == 0:
            out += symbol
            continue

        # a word is detected, correct it and push to output.
        c_word = correct_word(word)
        out += c_word + symbol
        logger.info('correct "%s" => "%s"', word, c_word)
        word = ''

    return out


def correct_word(word):
    d = english_dictionary()

    if d.check(word) is True:
        return word

    suggestions = d.suggest(word)
    if len(suggestions) == 0:
        return word

    return suggestions[0]


def english_dictionary():
    if english_dictionary._dict is None:
        d = os.path.dirname(__file__)
        extra_file = os.path.join(d, "eng_extra_dict")
        english_dictionary._dict = enchant.DictWithPWL("en_US", extra_file)

    return english_dictionary._dict


english_dictionary._dict = None


def pil_to_cv_image(image):
    rgb_image = image.convert('RGB')
    numpy_image = numpy.array(rgb_image)

    return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)


def graycale_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def cv_to_pil_image(image):
    t = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return PIL.Image.fromarray(t)


def is_english_symbol(symbol):
    code = ord(symbol)
    return (code >= 65 and code <= 90) or (code >= 97 and code <= 122)


def write_output(text, file):
    file.write(text)
    file.write('=========================================================\n')


command()
