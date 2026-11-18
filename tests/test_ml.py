from src.ml_extractor import canonical,label_lines,line_features,predict_fields,build_estimator

def test_weak_labeling():
    lines=["BOOK SHOP SDN BHD","NO 10 JALAN MAWAR","KUALA LUMPUR","DATE 25/12/2018","TOTAL 18.50","CASH 50.00"]
    gt={"company":"BOOK SHOP SDN BHD","address":"NO 10 JALAN MAWAR KUALA LUMPUR","date":"25/12/2018","total":"18.50"}
    labels=label_lines(lines,gt)
    assert labels[:5]==["COMPANY","ADDRESS","ADDRESS","DATE","TOTAL"]
    assert len(line_features(lines))==len(lines)
    assert canonical("RM 1,000.00")=="RM 1 000 00"

def test_empty():
    assert predict_fields([],object())=={"company":None,"date":None,"address":None,"total":None}

def test_fit_smoke():
    receipts=[
        (["ALPHA STORE","NO 12 JALAN MAWAR","DATE 25/12/2018","TOTAL 18.50","CASH 50.00"],
         {"company":"ALPHA STORE","address":"NO 12 JALAN MAWAR","date":"25/12/2018","total":"18.50"}),
        (["BETA MART","NO 20 JALAN DUA","DATE 10/01/2019","TOTAL 25.40","CASH 30.00"],
         {"company":"BETA MART","address":"NO 20 JALAN DUA","date":"10/01/2019","total":"25.40"})
    ]
    x,y=[],[]
    for lines,gt in receipts:
        x+=line_features(lines)
        y+=label_lines(lines,gt)
    model=build_estimator().fit(x,y)
    assert set(predict_fields(receipts[0][0],model))=={"company","date","address","total"}
