from typing import List

# Keyword mapping cho từng nguyên nhân
CAUSE_KEYWORDS = {
    "công việc": [
        "sếp", "deadline", "công việc", "làm việc", "văn phòng",
        "dự án", "họp", "overtime", "tăng ca", "đồng nghiệp",
        "khách hàng", "report", "kpi", "áp lực công việc"
    ],
    "giấc ngủ": [
        "ngủ", "mất ngủ", "thức khuya", "dậy sớm", "mệt",
        "buồn ngủ", "không ngủ được", "ngủ ít", "thiếu ngủ"
    ],
    "mối quan hệ": [
        "bạn bè", "người yêu", "gia đình", "bố", "mẹ",
        "anh", "chị", "em", "cãi nhau", "mâu thuẫn", "chia tay",
        "nhớ", "cô đơn", "bị bỏ rơi", "xa cách"
    ],
    "sức khỏe": [
        "đau", "bệnh", "ốm", "sốt", "đau đầu", "mệt mỏi",
        "không khỏe", "khám", "thuốc", "bệnh viện"
    ],
    "tài chính": [
        "tiền", "nợ", "chi tiêu", "lương", "tốn tiền",
        "hết tiền", "vay", "trả nợ", "tiết kiệm"
    ],
    "học tập": [
        "học", "thi", "điểm", "bài tập", "đồ án", "thầy",
        "cô", "trường", "lớp", "môn", "deadline bài"
    ],
    "thời tiết": [
        "nắng", "mưa", "lạnh", "nóng", "ẩm", "bão",
        "thời tiết", "trời"
    ],
}

def extract_causes(text: str) -> List[str]:
    text_lower = text.lower()
    found_causes = []
    
    for cause, keywords in CAUSE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                if cause not in found_causes:
                    found_causes.append(cause)
                break
    
    return found_causes

def extract_keywords(text: str) -> List[str]:
    """Trích xuất keywords quan trọng từ text"""
    all_keywords = []
    for keywords in CAUSE_KEYWORDS.values():
        for kw in keywords:
            if kw in text.lower() and len(kw) > 2:
                all_keywords.append(kw)
    return list(set(all_keywords))[:10]