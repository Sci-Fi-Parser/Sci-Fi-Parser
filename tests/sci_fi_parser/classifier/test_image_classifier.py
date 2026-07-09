from sci_fi_parser.classifier import image_classifier

IMAGE_PATH = "tests/test_materials/0a14bb795a27.jpg"


def test_classifier_with_default_parameters():
    classifier = image_classifier.ImageClassifier()
    im_path = IMAGE_PATH
    classifier_output = classifier.classify_image(im_path)
    assert classifier_output[0] in ["graphs_d", "graphs_h", "graphs_l", "graphs_s", "graphs_v", "graphs_val"]
    for v in classifier_output[1].values():
        assert 0<=v<=1
    assert abs(1-sum(classifier_output[1].values()))<0.001


def test_docling_model_output():
    classifier = image_classifier.DoclingClassifier()
    im_path = IMAGE_PATH
    classifier_output = classifier.classify_image(im_path)
    assert classifier_output[0] == "line_chart"
    assert (1 - classifier_output[1]["line_chart"]) < 0.01
