from django.db import models

# class Profile(models.Model):
#    name = models.CharField(max_length = 50)
#    picture = models.ImageField(upload_to = 'pictures')

#    class Meta:
#       db_table = "profile"

class FilesModel(models.Model):
    # file = models.FileField(upload_to='demo_files/%Y/%m/%d/')
    file = models.FileField(upload_to='demo_files/%Y/%m/%d/%H:%M:%S')
    date = models.DateTimeField(auto_now=True)
    name = models.CharField(max_length=30, blank=True, null=True)


class ContentModel(models.Model):
    filed_ids = models.CharField(max_length=20)
    name = models.CharField(max_length=20)
    sex = models.CharField(max_length=10)
    age = models.CharField(max_length=20)
    desc = models.CharField(max_length=200)
    file = models.ForeignKey(FilesModel)
