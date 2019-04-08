from django.conf.urls import url
from type import views

urlpatterns = [
     url(r'^$', views.index),
     url(r'^inquiryType', views.inquiry_type),
     url(r'^inquiryGoods', views.inquiry_goods),
     url(r'^inquiryAll', views.inquiry_all),
     url(r'^inquiryUnknownGoods', views.inquiry_uknown_goods),
     # url(r'^inquiryBatch', views.inquiry_batch),
]
