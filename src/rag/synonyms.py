import re

SYNONYMS = {
    "遠端桌面": ["VM", "虛擬機", "虛擬機器"],
    "軟體": ["程式"],
    "DDS": ["Digital Design Space"],
    "雲端檔案區": ["Remote Files", "遠端檔案區", "遠端區", "雲端區"],
    "檔案操作區": ["Workspace Files", "工作區", "個人工作區"],
    "工單": ["Task"],
    "審核者": ["Reviewer", "審查者", "審核人員", "審查人員"],
    "專案": ["Project"],
    "創建": ["Create", "建立", "開立"],
    "開始": ["Start", "啟動"],
    "繼續": ["Continue", "接續"],
    "完成": ["Resolve", "結案"],
    "結束": ["Close", "關閉"],
    "回溯": ["Reverse"],
    "回復": ["復原"],
    "上傳": ["Upload"],
    "下載": ["Download"],
    "刪除": ["Delete", "刪掉"],
    "重新命名": ["Rename", "改名", "更名"],
    "Lock": ["鎖定", "上鎖", "鎖住"],
    "Unlock": ["解鎖"],
    "被指派": ["Assigned"],
    "被拒絕": ["Rejected"],
    "編輯模式": ["Edit Mode"],
    "參考模式": ["Reference Mode"],
}

REVERSE_SYNONYMS = {}
for main, syns in SYNONYMS.items():
    for term in [main] + syns:
        REVERSE_SYNONYMS[term.lower()] = main

def strQ2B(ustring):
    r = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif 65281 <= inside_code <= 65374:
            inside_code -= 65248
        r += chr(inside_code)
    return r


def normalize_question(q: str) -> str:
    q = strQ2B(q)
    q = q.lower()
    for syn, main in REVERSE_SYNONYMS.items():
        if re.fullmatch(r"[a-zA-Z0-9 ]+", syn):
            q = re.sub(rf"\\b{re.escape(syn)}\\b", main, q)
        else:
            q = q.replace(syn, main)
    return q