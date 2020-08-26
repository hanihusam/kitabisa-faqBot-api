from django.http import JsonResponse
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from .engine.botrespon import response

# Create your views here.


@api_view(["POST"])
def bot_response(request):
    try:
        text = request.data['text']
        res = response(text)
        return JsonResponse(res, safe=False)
    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)
