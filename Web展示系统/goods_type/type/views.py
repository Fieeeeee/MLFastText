# encoding=utf8
from django.shortcuts import render
import pymysql
from predict import predict

db = pymysql.connect(host="127.0.0.1", user="root",
                     password="root", db="goods_type", port=3306)

cur = db.cursor()


def index(request):  # 首页
    return render(request, "index.html")


def inquiry_all(request):  # 查询全部
    sql = "select item_name,types from g_type1 limit 20"
    try:
        cur.execute(sql)  # 执行sql语句
        results = cur.fetchall()  # 获取查询的所有记录
    except Exception as e:
        raise e
    return render(request, "all_result_final.html", {"results": results})


def inquiry_type(request):  # 查询分类
    inquiry = request.POST.get("keyboard", None)
    new_inquiry = inquiry.split(',')
    new_inquiry = tuple(new_inquiry)
    print(new_inquiry)
    str = "('%s'"
    for i in range(len(new_inquiry) - 1):
        str = str + ",'%s'"
    na = str + ")"
    new_na = na % new_inquiry
    sql = "select item_name,types from g_type1 where ITEM_NAME in " + new_na + " limit 20"
    try:
        cur.execute(sql)  # 执行sql语句
        results = cur.fetchall()  # 获取查询的所有记录
        print(results)
    except Exception as e:
        raise e
    return render(request, "type_result_final.html", {"results": results})


def inquiry_goods(request):  # 查询商品
    inquiry = request.POST.get("keyboard", None)
    # print(inquiry)
    sql = "select item_name,types from g_type1 where types like '%%%s%%' limit 20" % inquiry
    try:
        cur.execute(sql)  # 执行sql语句
        results = cur.fetchall()  # 获取查询的所有记录
        print(results)
    except Exception as e:
        raise e
    return render(request, "goods_result_final.html", {"results": results})


def inquiry_uknown_goods(request):  # 查询未知分类商品
    inquiry = request.POST.get("keyboard", None)
    # 算法接口
    type = predict.fasttext_predic(inquiry)
    results = type
    print(results)
    return render(request, "unknown_goods_result.html", {"results": results})
