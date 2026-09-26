"""Date candidates for the Jev event_date shadow (retro#873) — code finds, Jev chooses.

The extractor's relative-date step is the one LLMs get wrong with confidence (retro#817: a
"Friday" walked to a Saturday), and the offline run for retro#873 found Haiku substituting
the publication date for a weekday-dated event. So the arithmetic stays in code: this module
finds date expressions (en/de/fr/es/ru/he/ar) and resolves each against the publication
date, and Jev only picks which candidate dates the event. Two rules make that a choice
rather than a guess:

- A weekday yields BOTH its last and its next occurrence (same day allowed in each), and a
  year-less "Feb. 28" both its past and upcoming occurrence: past vs future is a reading of
  the sentence, which is Jev's job.
- Vague periods stay periods: "last week" is that Mon–Sun, "in July" the whole month,
  "end of 2026" its December. The caller compares a date against the interval.

`dateparser` was measured and rejected for this (weekday sent to the wrong week, "Oct. 23"
split in two, the Hebrew day number lost). Offline, these candidates contain Haiku's date
for 82% of dated claims from a quote±2-sentence window, 88% from the whole article.
"""
from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

MONTHS = {}
for i, names in enumerate([
    "january jan januar janvier enero январь января ינואר يناير كانون الثاني",
    "february feb februar février febrero февраль февраля פברואר فبراير شباط",
    "march mar märz mars marzo март марта מרץ مارس آذار",
    "april apr avril abril апрель апреля אפריל أبريل نيسان",
    "may mai mayo май мая מאי مايو أيار",
    "june jun juni juin junio июнь июня יוני يونيو حزيران",
    "july jul juli juillet julio июль июля יולי يوليو تموز",
    "august aug août agosto август августа אוגוסט أغسطس آب",
    "september sep sept septembre septiembre сентябрь сентября ספטמבר سبتمبر أيلول",
    "october oct oktober octobre octubre октябрь октября אוקטובר أكتوبر تشرين الأول",
    "november nov novembre noviembre ноябрь ноября נובמבר نوفمبر تشرين الثاني",
    "december dec dez dezember décembre diciembre декабрь декабря דצמבר ديسمبر كانون الأول"], 1):
    for n in names.split():
        MONTHS.setdefault(n, i)
WEEKDAYS = {}
for i, names in enumerate([
    "monday montag lundi lunes понедельник שני الاثنين الإثنين",
    "tuesday dienstag mardi martes вторник שלישי الثلاثاء",
    "wednesday mittwoch mercredi miércoles среда среду רביעי الأربعاء الاربعاء",
    "thursday donnerstag jeudi jueves четверг חמישי الخميس",
    "friday freitag vendredi viernes пятница пятницу שישי الجمعة",
    "saturday samstag samedi sábado суббота субботу שבת السبت",
    "sunday sonntag dimanche domingo воскресенье ראשון الأحد الاحد"]):
    for n in names.split():
        WEEKDAYS[n] = i
RELATIVE = {"today": 0, "tonight": 0, "heute": 0, "aujourd'hui": 0, "hoy": 0, "сегодня": 0, "היום": 0, "הלילה": 0, "اليوم": 0,
            "yesterday": -1, "gestern": -1, "hier": -1, "ayer": -1, "вчера": -1, "אתמול": -1, "أمس": -1, "امس": -1, "البارحة": -1,
            "tomorrow": 1, "morgen": 1, "demain": 1, "mañana": 1, "завтра": 1, "מחר": 1, "غدا": 1, "غداً": 1}
HE_PREFIX = re.compile(r"^[ובלכהמש]{1,2}(?=[א-ת]{2,})")
MNAMES = "|".join(sorted((re.escape(m) for m in MONTHS), key=len, reverse=True))
MD = re.compile(rf"(?<![\w])({MNAMES})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?(?!\d)", re.I)
DM = re.compile(rf"(?<!\d)(\d{{1,2}})(?:st|nd|rd|th|\.|-?ה?ב?)?\s*(?:of\s+|de\s+|ב-?|ב)?({MNAMES})(?:,?\s+(\d{{4}}))?", re.I)
ISO = re.compile(r"(?<!\d)(20\d\d)-(\d\d)-(\d\d)(?!\d)")
NUM = re.compile(r"(?<![\d.])(\d{1,2})[./](\d{1,2})[./](20\d\d|\d\d)(?![\d.])")
WORD = re.compile(r"[\w'׳״-]+", re.U)

WEEK_REL = [(re.compile(p, re.I), k) for p, k in [
    (r"\blast week\b|\bletzte[rn]? woche\b|на прошлой неделе|בשבוע שעבר|השבוע שעבר|الأسبوع الماضي", -1),
    (r"\bthis week\b|\bdiese[rn]? woche\b|на этой неделе|השבוע(?! הבא| שעבר)|هذا الأسبوع", 0),
    (r"\bnext week\b|\bnächste[rn]? woche\b|на следующей неделе|בשבוע הבא|الأسبوع (?:المقبل|القادم)", 1)]]
MONTH_ONLY = re.compile(rf"(?<![\w])(?:(?:end|beginning|start|early|late|mid)[- ](?:of )?)?({MNAMES})\b\.?(?:\s+(20\d\d))?(?!\.?\s*\d)", re.I)
YEAR_END = re.compile(r"\b(?:end|close) of (20\d\d)\b|סוף (20\d\d)|конц[ае] (20\d\d)", re.I)

def _month_iv(y, m):
    return date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1])

def intervals(text: str, pub: date) -> list[tuple[date, date, str]]:
    """Every date expression in `text` as (start, end, verbatim expression), resolved against
    the publication date `pub`. Points have start == end. Deduplicated on (start, end)."""
    out = []
    def add(dt, expr, end=None):
        if dt and abs((dt - pub).days) < 3660:
            out.append((dt, end or dt, expr.strip()))
    def add_md(mon, day, expr):                     # year-less: both the last and the next occurrence
        for y in (pub.year - 1, pub.year, pub.year + 1):
            try: c = date(y, mon, day)
            except ValueError: continue
            if abs((c - pub).days) <= 366:
                add(c, expr + (" (past)" if c < pub else " (upcoming)" if c > pub else ""))
    for rx, kind in ((MD, "md"), (DM, "dm")):
        for m in rx.finditer(text):
            g = m.groups()
            mon, day = (MONTHS[g[0].lower()], int(g[1])) if kind == "md" else (MONTHS[g[1].lower()], int(g[0]))
            if not 1 <= day <= 31:
                continue
            try:
                add(date(int(g[2]), mon, day), m.group(0)) if g[2] else add_md(mon, day, m.group(0))
            except ValueError:
                pass
    for m in ISO.finditer(text):
        try: add(date(int(m[1]), int(m[2]), int(m[3])), m.group(0))
        except ValueError: pass
    for m in NUM.finditer(text):                           # day-first (IL/EU); US month-first is rare in this corpus
        y = int(m[3]); y = y + 2000 if y < 100 else y
        try: add(date(y, int(m[2]), int(m[1])), m.group(0))
        except ValueError: pass
    for w in WORD.finditer(text):
        tok = w.group(0).lower()
        base = HE_PREFIX.sub("", tok) if re.match(r"[א-ת]", tok) else tok
        for t in {tok, base}:
            if t in RELATIVE:
                add(pub + timedelta(days=RELATIVE[t]), w.group(0))
            if t in WEEKDAYS and t not in ("שני", "ראשון", "שבת") or (t in ("שני", "ראשון", "שבת") and re.search(r"ביום\s+" + t, text)):
                wd = WEEKDAYS[t]
                add(pub - timedelta(days=(pub.weekday() - wd) % 7), w.group(0) + " (past)")
                add(pub + timedelta(days=(wd - pub.weekday()) % 7), w.group(0) + " (upcoming)")
    for rx, k in WEEK_REL:
        for m in rx.finditer(text):
            mon0 = pub - timedelta(days=pub.weekday()) + timedelta(weeks=k)
            add(mon0, m.group(0), mon0 + timedelta(days=6))
    for m in MONTH_ONLY.finditer(text):
        mon = MONTHS[m.group(1).lower()]
        if m.group(1).lower() in ("may", "mar", "jan", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "mai", "hier") and not m.group(2) \
                and not re.match(r"(?i)(end|beginning|start|early|late|mid)", m.group(0)) and m.group(1)[0].islower():
            continue                                 # "may" the verb, etc.
        if m.group(2):
            a, b = _month_iv(int(m.group(2)), mon); add(a, m.group(0), b)
        else:
            for y in (pub.year - 1, pub.year, pub.year + 1):
                a, b = _month_iv(y, mon)
                if b >= pub - timedelta(days=366) and a <= pub + timedelta(days=366):
                    add(a, m.group(0) + (" (past)" if b < pub else " (upcoming)" if a > pub else " (current)"), b)
    for m in YEAR_END.finditer(text):
        y = int(next(g for g in m.groups() if g)); add(date(y, 12, 1), m.group(0), date(y, 12, 31))
    seen, res = set(), []
    for a, b, e in out:
        if (a, b) not in seen:
            seen.add((a, b)); res.append((a, b, e))
    return res
