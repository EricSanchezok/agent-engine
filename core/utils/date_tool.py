from datetime import datetime, timedelta
import re

def get_last_week_range() -> str:
    today = datetime.today()
    
    last_monday = today - timedelta(days=today.weekday() + 7)
    
    last_sunday = last_monday + timedelta(days=6)
    
    start_date = last_monday.strftime("%Y%m%d")
    end_date = last_sunday.strftime("%Y%m%d")
    
    return f"{start_date}-{end_date}"

def get_weekday(date_str: str) -> str:
    # 支持多种日期格式
    formats_to_try = [
        "%Y-%m-%d",  # 2025-08-03
        "%Y%m%d",    # 20250803
        "%Y/%m/%d",  # 2025/08/03
    ]
    
    for fmt in formats_to_try:
        try:
            date_obj = datetime.strptime(date_str, fmt)
            weekday_index = date_obj.weekday()
            if weekday_index == 0:
                return "周一"
            elif weekday_index == 1:
                return "周二"
            elif weekday_index == 2:
                return "周三"
            elif weekday_index == 3:
                return "周四"
            elif weekday_index == 4:
                return "周五"
            elif weekday_index == 5:
                return "周六"
            elif weekday_index == 6:
                return "周日"
        except ValueError:
            continue
    
    # 如果所有格式都失败，返回默认值
    return "未知"

class DateFormatter:
    def __init__(self, target_format="%Y%m%d"):
        self.target_format = target_format
        self.date_patterns = {
            "YYYYMMDD": r"\d{4}[01]\d[0-3]\d",  # YYYYMMDD格式正则
            "YYYY-MM-DD": r"\d{4}-[01]\d-[0-3]\d",  # ISO格式正则
            "DD/MM/YYYY": r"[0-3]\d/[01]\d/\d{4}",  # 欧洲格式正则
            "MM/DD/YYYY": r"[01]\d/[0-3]\d/\d{4}",  # 美国格式正则
            "YYYY.MM.DD": r"\d{4}\.[01]\d\.[0-3]\d",  # 点分隔格式
            "YYYY年MM月DD日": r"\d{4}年[01]\d月[0-3]\d日",  # 中文格式
            "timestamp": r"\d{10,13}",  # 时间戳格式（10-13位）
        }
    
    def is_valid_date(self, date_str):
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            pass
        
        for fmt in ("%Y%m%d", "%d/%m/%Y", "%m/%d/%Y", "%Y.%m.%d", "%Y年%m月%d日"):
            try:
                datetime.strptime(date_str, fmt)
                return True
            except:
                continue
        
        return False
    
    def normalize_date(self, date_str):
        if re.fullmatch(self.date_patterns["timestamp"], date_str):
            try:
                ts_length = len(date_str)
                timestamp = int(date_str)
                if ts_length > 10: timestamp = timestamp / 1000.0
                return datetime.fromtimestamp(timestamp).strftime(self.target_format)
            except:
                pass
        
        for fmt_name, pattern in self.date_patterns.items():
            if fmt_name == "timestamp": continue
            
            if re.fullmatch(pattern, date_str):
                if fmt_name == "YYYYMMDD":
                    return date_str
                
                try:
                    fmt_map = {
                        "YYYY-MM-DD": "%Y-%m-%d",
                        "DD/MM/YYYY": "%d/%m/%Y",
                        "MM/DD/YYYY": "%m/%d/%Y",
                        "YYYY.MM.DD": "%Y.%m.%d",
                        "YYYY年MM月DD日": "%Y年%m月%d日",
                    }
                    return datetime.strptime(date_str, fmt_map[fmt_name]).strftime(self.target_format)
                except:
                    continue
        
        try:
            return self._parse_and_convert(date_str)
        except ValueError:
            # 无法识别的格式，保持原样
            return date_str
    
    def _parse_and_convert(self, date_str):
        separators = ["-", "/", ".", "\\", "年", "月", "日", " ", ":"]
        for sep in separators:
            parts = date_str.split(sep)
            if len(parts) in (3, 4):
                try:
                    formats_to_try = [
                        ("%Y", "%m", "%d"),  # YMD
                        ("%d", "%m", "%Y"),  # DMY
                        ("%m", "%d", "%Y"),  # MDY
                        ("%Y", "%m", "%d", "%H:%M:%S")  # 带时间的格式
                    ]
                    
                    for fmt_combo in formats_to_try:
                        try:
                            date_fmt = sep.join(fmt_combo)
                            dt = datetime.strptime(date_str, date_fmt)
                            return dt.strftime(self.target_format)
                        except:
                            continue
                except:
                    continue
                    
        date_str_clean = re.sub(r"[^\d]", "", date_str)
        if len(date_str_clean) == 8:
            try:
                datetime.strptime(date_str_clean, "%Y%m%d")
                return date_str_clean
            except:
                pass
            
        if re.search(r"\d{6,8}", date_str):
            matched = re.search(r"(\d{4})[^\d]*?(\d{2})[^\d]*?(\d{2})", date_str)
            if matched:
                try:
                    year, month, day = map(int, matched.groups())
                    dt = datetime(year, month, day)
                    return dt.strftime(self.target_format)
                except ValueError:
                    pass
        
        return date_str
    
    def reformat_dates_in_query(self, query):
        pattern = r"""
            (?P<prefix>[a-zA-Z]*:)?       # 字段名
            (                              # 整个日期表达式（可选）
            \[                             # 开始括号
            (?P<start>[^]]*?)              # 开始日期
            (?:\s+TO\s+(?P<end>[^]]*?))?   # 结束日期（可选）
            \]                             # 结束括号
            )
        """
        
        def replace_date(match):
            prefix = match.group("prefix") or ""
            start_date = match.group("start")
            end_date = match.group("end")
            
            formatted_start = self.normalize_date(start_date)
            
            formatted_end = self.normalize_date(end_date) if end_date else None
            
            if formatted_end:
                date_expression = f"[{formatted_start} TO {formatted_end}]"
            else:
                date_expression = f"[{formatted_start}]"
            
            return f"{prefix}{date_expression}"
        
        return re.sub(pattern, replace_date, query, flags=re.VERBOSE)
