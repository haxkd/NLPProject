
from . import views
from django.contrib import admin
from django.urls import path

urlpatterns = [
    path('',views.index,name='index'),
    path('semantic/',views.semantic,name='semantic'),
    path('sentence/',views.sentence,name='sentence'),
    path('essay/',views.essay,name='essay'),
    path('modules/',views.modules,name='modules')
]
