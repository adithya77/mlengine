from django.http import HttpResponse

from Engine.Algos.Classifiers import Util
from Engine.Algos.Classifiers.NavieBayes import NaviesBayes
from urllib.parse import unquote

nb = NaviesBayes()


def train(request):
    print("In Training view")
    nb.train()
    return HttpResponse("Training Finished!! Go Die")


def predict(request):
    data = request.GET['data']
    data = unquote(data)
    print(data)
    data = Util.extract_features_predict(data, nb.get_dictionary())
    print(data)
    val = nb.predict(data)
    print("Predicted {0}, is {1}".format(data, val[0]))
    if val[0] == 0.0:
        is_spam = 'not a spam'
    else:
        is_spam = 'spam'
    return HttpResponse(is_spam)


