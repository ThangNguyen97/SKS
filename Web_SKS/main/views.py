from django.shortcuts import render, HttpResponse
from django.views import View
from django.http import JsonResponse

import requests
import json

import base64
# Create your views here.


class KeywordView(View):
    def get(self, request):
        return render(request, 'voice/keyword.html')
    def post(self, request):
        print("----------------------------------")
        tab_status1 = request.POST.get("tab_status1")
        print("tab_status1: ", tab_status1)
        if tab_status1 == "block":
            file = request.FILES.get("upload_file")
            model = request.POST.get("model")
            name_audio = str(file)
        else:
            file = request.FILES.get("audio_data")
            model = request.POST.get("model")
            name_audio = str(file) + ".wav"
        # print("file: ", file)
        print("model: ", model)
        convert_model ={"model1": "KWT", "model2": "BCR"}
        model = convert_model[model]
        print("model: ", model)

        print("name_audio: ", name_audio)
        audio = file.open("rb").read()

        url = "http://172.22.10.2:35000/keyword_spotting"

        payload = {'model': model}
        files = [
            ('file', (name_audio, audio, 'audio/wav'))
        ]
        headers = {}

        response = requests.request("POST", url, headers=headers, data=payload, files=files)

        # print(response.text)

        content = response.text
        res = json.loads(content)
        print(res)

        return JsonResponse(res)
        # return HttpResponse("oke")
