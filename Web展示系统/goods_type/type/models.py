from django.db import models

# Create your models here.


class Goods(models.Model):
    ITEM_NAME = models.CharField(max_length=255)
    TYPE = models.CharField(max_length=255)

