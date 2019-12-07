# -*- coding:utf-8 -*-
from django.forms import ModelForm
from .model import  FilesModel, CommentModel


class UploadFileForm(ModelForm):
    class Meta:
        model = FilesModel
        fields = ('file',)



class CommentForm(ModelForm):
    class Meta:
        model = CommentModel
        fields = '__all__'
