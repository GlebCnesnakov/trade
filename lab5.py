import sys
import datetime
from dataclasses import dataclass
from typing import List, Optional

import requests
import pandas as pd
import numpy as np

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QComboBox, QWidgetAction
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
import mplfinance as mpf


# URL ISS МОЕХ для свечей по акциям
MOEX_CANDLES_URL = (
    "https://iss.moex.com/iss/engines/stock/markets/shares/securities/{secid}/candles.json"
)


def fetch_candles_from_moex(
    secid: str = "SBER",
    days: int = 365,
    interval: int = 24,
) -> pd.DataFrame:
    """
    Загрузка свечей с МОЕХ через ISS API.
    Возвращает DataFrame с колонками [Open, High, Low, Close, Volume]
    и индексом-датой (begin).
    """
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=days)

    params = {
        "from": start_date.strftime("%Y-%m-%d"),
        "till": today.strftime("%Y-%m-%d"),
        "interval": interval,  # 24 = дневные свечи
        "start": 0,
    }

    url = MOEX_CANDLES_URL.format(secid=secid)

    # отдельная сессия, отключаем системные прокси
    session = requests.Session()
    session.trust_env = False

    resp = session.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    candles = data.get("candles", {})
    cols = candles.get("columns", [])
    rows = candles.get("data", [])

    if not rows:
        raise RuntimeError("От МОЕХ не пришли данные по свечам")

    df = pd.DataFrame(rows, columns=cols)

    if "begin" not in df.columns:
        raise RuntimeError("Неожиданная структура ответа МОЕХ (нет поля 'begin')")

    dt = pd.to_datetime(df["begin"])
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert(None)
    df["begin"] = dt
    df.set_index("begin", inplace=True)
    df.sort_index(inplace=True)

    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }

    for old in rename_map.keys():
        if old not in df.columns:
            raise RuntimeError(f"В ответе МОЕХ нет колонки '{old}'")

    df = df[list(rename_map.keys())]
    df.rename(columns=rename_map, inplace=True)
    df = df.astype(float)

    return df


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Простейший RSI по классической формуле с SMA."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ---------------- ПАТТЕРНЫ ---------------- #
def detect_hanging_man(df: pd.DataFrame) -> List[pd.Timestamp]:
    """Примитивный детектор паттерна 'висельник' (hanging man)."""
    opens = df["Open"]
    highs = df["High"]
    lows = df["Low"]
    closes = df["Close"]

    dates = []
    for i in range(1, len(df)):
        o = opens.iloc[i]
        h = highs.iloc[i]
        l = lows.iloc[i]
        c = closes.iloc[i]

        body = abs(c - o)
        full_range = h - l
        if full_range == 0:
            continue

        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l

        # маленькое тело, длинная нижняя тень, короткая верхняя
        if body / full_range < 0.3 and lower_shadow > body * 2 and upper_shadow < body * 0.4:
            start = max(0, i - 4)
            prev_mean = closes.iloc[start:i].mean()
            if c > prev_mean:
                dates.append(df.index[i])

    return dates


def detect_bearish_harami_cross(df: pd.DataFrame) -> List[pd.Timestamp]:
    """Примитивный детектор паттерна 'медвежий крест харами'."""
    o = df["Open"]
    h = df["High"]
    l = df["Low"]
    c = df["Close"]

    result = []
    for i in range(1, len(df)):
        o1, h1, l1, c1 = o.iloc[i - 1], h.iloc[i - 1], l.iloc[i - 1], c.iloc[i - 1]
        o2, h2, l2, c2 = o.iloc[i], h.iloc[i], l.iloc[i], c.iloc[i]

        body1 = abs(c1 - o1)
        range1 = h1 - l1
        body2 = abs(c2 - o2)
        range2 = h2 - l2

        if range1 == 0 or range2 == 0:
            continue

        # сильная бычья свеча
        if not (c1 > o1 and body1 / range1 > 0.6):
            continue

        # doji — почти нет тела
        if not (body2 / range2 < 0.1):
            continue

        min_body1 = min(o1, c1)
        max_body1 = max(o1, c1)

        high2_in_body1 = h2 <= max_body1
        low2_in_body1 = l2 >= min_body1

        if high2_in_body1 and low2_in_body1 and c2 < c1:
            result.append(df.index[i])

    return result


def detect_diamond(df: pd.DataFrame, window: int = 20) -> List[pd.Timestamp]:
    """Очень грубый детектор паттерна 'бриллиант' (diamond top)."""
    closes = df["Close"]
    res = []
    n = len(df)
    half = window // 2
    if n < window:
        return res

    for i in range(half, n - half):
        segment = df.iloc[i - half : i + half + 1]
        widths = segment["High"] - segment["Low"]
        if widths.isna().any():
            continue

        center_width = widths.iloc[half]
        left_mean = widths.iloc[:half].mean()
        right_mean = widths.iloc[half + 1 :].mean()

        if not (center_width > left_mean * 1.1 and center_width > right_mean * 1.1):
            continue

        center_close = closes.iloc[i]
        local_max = segment["Close"].max()
        if center_close < local_max * 0.995:
            continue

        pre_start = max(0, i - window)
        if closes.iloc[i] <= closes.iloc[pre_start]:
            continue

        res.append(df.index[i])

    return res


def detect_romb(df: pd.DataFrame, window: int = 20) -> List[pd.Timestamp]:
    """Очень грубый детектор 'ромб' (diamond bottom)."""
    closes = df["Close"]
    res = []
    n = len(df)
    half = window // 2
    if n < window:
        return res

    for i in range(half, n - half):
        segment = df.iloc[i - half : i + half + 1]
        widths = segment["High"] - segment["Low"]
        if widths.isna().any():
            continue

        center_width = widths.iloc[half]
        left_mean = widths.iloc[:half].mean()
        right_mean = widths.iloc[half + 1 :].mean()

        if not (center_width > left_mean * 1.1 and center_width > right_mean * 1.1):
            continue

        center_close = closes.iloc[i]
        local_min = segment["Close"].min()
        if center_close > local_min * 1.005:
            continue

        pre_start = max(0, i - window)
        if closes.iloc[i] >= closes.iloc[pre_start]:
            continue

        res.append(df.index[i])

    return res


def detect_ascending_wedge(df: pd.DataFrame, window: int = 20, step: int = 5) -> List[pd.Timestamp]:
    """
    Детектор 'восходящего клина' с улучшенными параметрами.
    """
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    n = len(df)
    res_indices: List[int] = []

    if n < window + 5:
        return []

    x_full = np.arange(n)

    for start in range(0, n - window, step):
        end = start + window
        segment_x = x_full[start:end]
        seg_high = highs[start:end]
        seg_low = lows[start:end]
        seg_close = closes[start:end]

        # Линейная регрессия по максимумам и минимумам
        k_high, b_high = np.polyfit(segment_x, seg_high, 1)
        k_low, b_low = np.polyfit(segment_x, seg_low, 1)

        # Оба тренда должны быть восходящими
        if k_high <= 0 or k_low <= 0:
            continue

        # Увеличим фильтр для роста наклона нижней линии
        if k_low <= k_high * 1.1:
            continue

        avg_price = seg_close.mean()
        if avg_price == 0:
            continue

        # Нормируем наклоны к уровню цены
        norm_k_high = k_high / avg_price
        norm_k_low = k_low / avg_price

        # Нижняя линия должна быть заметно круче верхней
        if not (norm_k_low > norm_k_high * 0.9):  # Можно увеличить коэффициент для жесткости
            continue

        width_start = seg_high[0] - seg_low[0]
        width_end = seg_high[-1] - seg_low[-1]
        if width_start <= 0 or width_end <= 0:
            continue

        # Клин должен сужаться
        if width_end > width_start * 0.95:  # Уменьшаем предел сужения для точности
            continue

        # Общий рост цены за окно (5%+)
        rise = (seg_close[-1] / seg_close[0]) - 1.0
        if rise < 0.05:
            continue

        # Небольшой фильтр "красоты": High/Low должны быть более-менее линейными
        x0 = np.arange(window)
        corr_high = np.corrcoef(x0, seg_high)[0, 1]
        corr_low = np.corrcoef(x0, seg_low)[0, 1]
        if corr_high < 0.7 or corr_low < 0.7:  # Увеличиваем минимальную корреляцию
            continue

        center_idx = start + window // 2
        res_indices.append(center_idx)

    # --- дедупликация сигналов, которые почти рядом ---
    res_indices.sort()
    final_indices: List[int] = []
    min_dist = window  # Минимальная длина окна между клинами

    last = -10**9
    for idx in res_indices:
        if not final_indices or idx - last >= min_dist:
            final_indices.append(idx)
            last = idx

    return [df.index[i] for i in final_indices]
@dataclass
class PatternResults:
    hanging_man: List[pd.Timestamp]
    diamond: List[pd.Timestamp]
    romb: List[pd.Timestamp]
    bearish_harami_cross: List[pd.Timestamp]
    ascending_wedge: List[pd.Timestamp]


def compute_all_patterns(df: pd.DataFrame) -> PatternResults:
    return PatternResults(
        hanging_man=detect_hanging_man(df),
        diamond=detect_diamond(df),
        romb=detect_romb(df),
        bearish_harami_cross=detect_bearish_harami_cross(df),
        ascending_wedge=detect_ascending_wedge(df),
    )


# ---------------- ИШИМОКУ ---------------- #
def compute_ichimoku(df: pd.DataFrame) -> dict:
    """Вычисление линий Ишимоку."""
    df_sorted = df.sort_index()

    def _midpoint(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
        """Средняя точка диапазона за окно с полным количеством данных."""
        rolling_high = high.rolling(window=window, min_periods=window).max()
        rolling_low = low.rolling(window=window, min_periods=window).min()
        return (rolling_high + rolling_low) / 2

    tenkan_sen = _midpoint(df_sorted["High"], df_sorted["Low"], 9)
    kijun_sen = _midpoint(df_sorted["High"], df_sorted["Low"], 26)

    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    senkou_span_b = _midpoint(df_sorted["High"], df_sorted["Low"], 52).shift(26)
    chikou_span = df_sorted["Close"].shift(-26)

    return {
        "Tenkan-sen": tenkan_sen,
        "Kijun-sen": kijun_sen,
        "Senkou Span A": senkou_span_a,
        "Senkou Span B": senkou_span_b,
        "Chikou Span": chikou_span,
    }


def compute_alligator(df: pd.DataFrame) -> dict:
    """Индикатор Аллигатор (челюсти/зубы/губы) по медианной цене."""
    df_sorted = df.sort_index()
    median_price = (df_sorted["High"] + df_sorted["Low"]) / 2

    def _smoothed(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    jaw = _smoothed(median_price, 13).shift(8)
    teeth = _smoothed(median_price, 8).shift(5)
    lips = _smoothed(median_price, 5).shift(3)

    return {"Jaw": jaw, "Teeth": teeth, "Lips": lips}


# ---------------- GUI ---------------- #

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(10, 6))
        self.figure = fig
        self.axes_main = fig.add_axes([0.05, 0.42, 0.9, 0.55])
        self.axes_alligator = fig.add_axes(
            [0.05, 0.23, 0.9, 0.15], sharex=self.axes_main
        )
        self.axes_rsi = fig.add_axes([0.05, 0.05, 0.9, 0.13], sharex=self.axes_main)
        super().__init__(fig)
        self.setParent(parent)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MOEX SBER Pattern Finder")

        self.canvas = MplCanvas(self)

        # === центральный виджет: canvas + toolbar ===
        central_widget = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)

        self.setCentralWidget(central_widget)
        # === конец изменения для зума ===

        self.df: Optional[pd.DataFrame] = None
        self.patterns: Optional[PatternResults] = None
        self.display_ichimoku = False  # Ишемоку по умолчанию не показывать
        self.show_rsi = True  # По умолчанию отображается RSI

        # Новый атрибут для хранения выбранной компании
        self.selected_company = "SBER"  # Инициализируем его значением по умолчанию

        self._build_menu()

        self.statusBar().showMessage("Загрузка данных с МОЕХ...")
        QtCore.QTimer.singleShot(100, self.load_data_and_draw)

        self.showMaximized()

    def _build_menu(self):
        menubar = self.menuBar()

        data_menu = menubar.addMenu("Данные")
        reload_action = QtWidgets.QAction("Перезагрузить данные", self)
        reload_action.triggered.connect(self.load_data_and_draw)
        data_menu.addAction(reload_action)

        # Меню для выбора компании
        company_menu = menubar.addMenu("Компания")

        # Создаем QComboBox для выбора компании
        self.company_selector = QComboBox(self)
        self.company_selector.addItem("SBER")
        self.company_selector.addItem("GAZP")
        self.company_selector.addItem("AFLT")
        self.company_selector.addItem("YNDX")
        self.company_selector.addItem("TATN")
        self.company_selector.currentTextChanged.connect(self.on_company_change)

        # Создаем QWidgetAction для добавления QComboBox в меню
        company_action = QWidgetAction(self)
        company_action.setDefaultWidget(self.company_selector)
        company_menu.addAction(company_action)

        # Меню для выбора индикаторов
        indicator_menu = menubar.addMenu("Индикаторы")

        # Включить Ишимоку
        act_ichimoku = QtWidgets.QAction("Показать Ишимоку", self)
        act_ichimoku.setCheckable(True)
        act_ichimoku.triggered.connect(self.toggle_ichimoku)
        indicator_menu.addAction(act_ichimoku)

        # Включить RSI
        act_rsi = QtWidgets.QAction("Показать RSI", self)
        act_rsi.setCheckable(True)
        act_rsi.triggered.connect(self.toggle_rsi)
        indicator_menu.addAction(act_rsi)

        # Паттерны
        pattern_menu = menubar.addMenu("Паттерны")

        act_hanging = QtWidgets.QAction("Найти 'Висельник'", self)
        act_hanging.triggered.connect(lambda: self.highlight_pattern("hanging_man"))
        pattern_menu.addAction(act_hanging)

        act_diamond = QtWidgets.QAction("Найти 'Бриллиант'", self)
        act_diamond.triggered.connect(lambda: self.highlight_pattern("diamond"))
        pattern_menu.addAction(act_diamond)

        act_romb = QtWidgets.QAction("Найти 'Ромб'", self)
        act_romb.triggered.connect(lambda: self.highlight_pattern("romb"))
        pattern_menu.addAction(act_romb)

        act_harami = QtWidgets.QAction("Найти 'Медвежий крест харами'", self)
        act_harami.triggered.connect(lambda: self.highlight_pattern("bearish_harami_cross"))
        pattern_menu.addAction(act_harami)

        act_wedge = QtWidgets.QAction("Найти 'Восходящий клин'", self)
        act_wedge.triggered.connect(lambda: self.highlight_pattern("ascending_wedge"))
        pattern_menu.addAction(act_wedge)

        pattern_menu.addSeparator()

        act_all = QtWidgets.QAction("Показать все паттерны", self)
        act_all.triggered.connect(lambda: self.highlight_pattern("all"))
        pattern_menu.addAction(act_all)

        act_clear = QtWidgets.QAction("Очистить выделение", self)
        act_clear.triggered.connect(lambda: self.highlight_pattern(None))
        pattern_menu.addAction(act_clear)

    def toggle_ichimoku(self):
        """Переключаем отображение Ишимоку"""
        self.display_ichimoku = not self.display_ichimoku
        self.load_data_and_draw()

    def toggle_rsi(self):
        """Переключаем отображение RSI"""
        self.show_rsi = not self.show_rsi
        self.load_data_and_draw()

    def on_company_change(self, company_name: str):
        """Обработчик выбора компании из выпадающего списка."""
        self.selected_company = company_name
        self.load_data_and_draw()  # Перезагружаем данные для выбранной компании

    def load_data_and_draw(self):
        """Загрузка данных для выбранной компании и отрисовка графика."""
        try:
            self.statusBar().showMessage(f"Загрузка данных для {self.selected_company} с МОЕХ...")
            df = fetch_candles_from_moex(self.selected_company, days=365, interval=24)
            df["RSI"] = compute_rsi(df["Close"], period=14)
            self.df = df
            self.patterns = compute_all_patterns(df)
            self.statusBar().showMessage(
                "Данные загружены. Найдено: висельник={} бриллиант={} ромб={} медвежий крест харами={}".format(
                    len(self.patterns.hanging_man),
                    len(self.patterns.diamond),
                    len(self.patterns.romb),
                    len(self.patterns.bearish_harami_cross),
                )
            )
            self.draw_chart()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Ошибка",
                f"Не удалось загрузить данные с МОЕХ: {exc}",
            )
            self.statusBar().showMessage("Ошибка загрузки данных")

    def draw_chart(self, highlight_key: Optional[str] = None):
        """Рисуем свечи + RSI + Аллигатор/Ишимоку. При необходимости подсвечиваем паттерны."""
        if self.df is None:
            return

        # последние 180 баров
        df = self.df.tail(180).copy()

        ax_main = self.canvas.axes_main
        ax_alligator = self.canvas.axes_alligator
        ax_rsi = self.canvas.axes_rsi

        ax_main.clear()
        ax_alligator.clear()
        ax_rsi.clear()

        # свечи
        mpf.plot(
            df[["Open", "High", "Low", "Close", "Volume"]],
            type="candle",
            ax=ax_main,
            volume=False,
            style="yahoo",
            show_nontrading=False,
        )

        ax_main.set_ylabel("Цена")
        ax_main.grid(True)

        xleft, xright = ax_main.get_xlim()
        x_index = df.index

        # RSI
        rsi = df["RSI"]
        ax_rsi.plot(x_index, rsi.values, linewidth=1)
        ax_rsi.axhline(70, linestyle="--", linewidth=0.8)
        ax_rsi.axhline(30, linestyle="--", linewidth=0.8)
        ax_rsi.set_ylabel("RSI")
        ax_rsi.grid(True)
        ax_rsi.set_xlim(xleft, xright)

        # Аллигатор
        alligator = pd.DataFrame(compute_alligator(df)).reindex(df.index)
        ax_alligator.plot(x_index, alligator["Jaw"], label="Jaw (13, +8)", color="blue")
        ax_alligator.plot(
            x_index, alligator["Teeth"], label="Teeth (8, +5)", color="red"
        )
        ax_alligator.plot(
            x_index, alligator["Lips"], label="Lips (5, +3)", color="green"
        )
        ax_alligator.set_ylabel("Alligator")
        ax_alligator.grid(True)
        ax_alligator.legend(loc="upper left", fontsize=8)
        ax_alligator.set_xlim(xleft, xright)

        # Отображаем Ишимоку, если выбран
        if self.display_ichimoku:
            ichimoku = compute_ichimoku(df)
            ichimoku_df = pd.DataFrame(ichimoku).reindex(df.index)

            tenkan = ichimoku_df["Tenkan-sen"]
            kijun = ichimoku_df["Kijun-sen"]
            span_a = ichimoku_df["Senkou Span A"]
            span_b = ichimoku_df["Senkou Span B"]
            chikou = ichimoku_df["Chikou Span"]

            ax_main.plot(df.index, tenkan, label="Tenkan-sen", color="blue", linewidth=1.1)
            ax_main.plot(df.index, kijun, label="Kijun-sen", color="red", linewidth=1.1)

            valid_mask = ~(span_a.isna() | span_b.isna())
            bull_mask = valid_mask & (span_a >= span_b)
            bear_mask = valid_mask & (span_a < span_b)

            ax_main.fill_between(
                df.index,
                span_a,
                span_b,
                where=bull_mask,
                color="#F0B27A",
                alpha=0.35,
                interpolate=True,
                label="Kumo (A>B)",
            )
            ax_main.fill_between(
                df.index,
                span_a,
                span_b,
                where=bear_mask,
                color="#BB8FCE",
                alpha=0.35,
                interpolate=True,
                label="Kumo (B>A)",
            )

            ax_main.plot(df.index, span_a, label="Senkou Span A", color="green", linewidth=1)
            ax_main.plot(df.index, span_b, label="Senkou Span B", color="orange", linewidth=1)
            ax_main.plot(df.index, chikou, label="Chikou Span", color="purple", linewidth=1)

            ax_main.legend(loc="upper left")

        # ---- отфильтрованные по видимому участку паттерны ----
        visible_index = df.index

        def filt(lst: List[pd.Timestamp]) -> List[pd.Timestamp]:
            return [d for d in lst if d in visible_index]

        visible_patterns = PatternResults(
            hanging_man=filt(self.patterns.hanging_man),
            diamond=filt(self.patterns.diamond),
            romb=filt(self.patterns.romb),
            bearish_harami_cross=filt(self.patterns.bearish_harami_cross),
            ascending_wedge=filt(self.patterns.ascending_wedge),
        )

        # --- убираем старые figure.text (если были) ---
        for t in list(self.canvas.figure.texts):
            t.remove()

        # строка со счётчиком ПАТТЕРНОВ (только по видимому отрезку)
        txt = (
            f"Видимый участок: "
            f"Висельник: {len(visible_patterns.hanging_man)}  |  "
            f"Бриллиант: {len(visible_patterns.diamond)}  |  "
            f"Ромб: {len(visible_patterns.romb)}  |  "
            f"Медвежий крест харами: {len(visible_patterns.bearish_harami_cross)}  |  "
            f"Восходящий клин: {len(visible_patterns.ascending_wedge)}"
        )
        self.canvas.figure.text(
            0.01,
            0.03,
            txt,
            ha="left",
            va="bottom",
            fontsize=9,
        )

        # подсветка паттернов
        if highlight_key is not None:
            self._draw_patterns(df, highlight_key, ax_main, visible_patterns)

        self.canvas.figure.autofmt_xdate()
        self.canvas.draw()

    def _draw_patterns(self, df: pd.DataFrame, key: str, ax_main, patterns: PatternResults):
        """
        Рисуем маркеры и линии для паттернов.
        """
        def date_indices(dates: List[pd.Timestamp]) -> List[int]:
            idx_list = []
            for d in dates:
                if d in df.index:
                    idx_list.append(df.index.get_loc(d))
            return idx_list

        def scatter_dates(indices, color, marker, label):
            if not indices:
                return
            xs = np.array(indices, dtype=float)
            ys = df["High"].iloc[indices].values * 1.001
            ax_main.scatter(
                xs,
                ys,
                marker=marker,
                s=60,
                label=label,
                color=color,
                zorder=5,
            )

        def draw_vertical_for_indices(indices, color):
            for i in indices:
                low = df["Low"].iloc[i]
                high = df["High"].iloc[i]
                ax_main.vlines(i, low, high, colors=color, linestyles="--", linewidth=1)

        def draw_box_for_indices(indices, color, half_window=10):
            n = len(df)
            for i in indices:
                start = max(0, i - half_window)
                end = min(n - 1, i + half_window)
                seg = df.iloc[start : end + 1]
                top = seg["High"].max()
                bottom = seg["Low"].min()
                ax_main.hlines(top, start, end, colors=color, linestyles="--", linewidth=1.2)
                ax_main.hlines(bottom, start, end, colors=color, linestyles="--", linewidth=1.2)

        def draw_wedge_for_indices(indices, color, window=30):
            n = len(df)
            for center in indices:
                start = max(0, center - window // 2)
                end = min(n - 1, center + window // 2)
                x = np.arange(start, end + 1)
                seg = df.iloc[start : end + 1]
                highs = seg["High"].values
                lows = seg["Low"].values

                if len(x) < 3:
                    continue

                # Линейная регрессия для верхней и нижней линии
                k_high, b_high = np.polyfit(x, highs, 1)
                k_low, b_low = np.polyfit(x, lows, 1)

                # Проверка, что обе линии имеют восходящий тренд
                if k_high <= 0 or k_low <= 0:
                    continue

                # Нормируем наклоны
                avg_price = seg["Close"].mean()
                norm_k_high = k_high / avg_price
                norm_k_low = k_low / avg_price

                # Линия должна быть сужающейся, поэтому нижняя линия круче
                if norm_k_low > norm_k_high:
                    ax_main.plot(x, k_high * x + b_high, color=color, linestyle="--", linewidth=1.5)
                    ax_main.plot(x, k_low * x + b_low, color=color, linestyle="--", linewidth=1.5)


        # индексы
        idx_hanging = date_indices(patterns.hanging_man)
        idx_diamond = date_indices(patterns.diamond)
        idx_romb = date_indices(patterns.romb)
        idx_harami = date_indices(patterns.bearish_harami_cross)
        idx_wedge = date_indices(patterns.ascending_wedge)

        if key == "all":
            scatter_dates(idx_hanging, "red", "v", "Висельник")
            scatter_dates(idx_diamond, "blue", "D", "Бриллиант")
            scatter_dates(idx_romb, "green", "s", "Ромб")
            scatter_dates(idx_harami, "black", "x", "Медвежий крест харами")
            scatter_dates(idx_wedge, "orange", "^", "Восходящий клин")

            draw_vertical_for_indices(idx_hanging, "red")
            draw_vertical_for_indices(idx_harami, "black")
            draw_box_for_indices(idx_diamond, "blue")
            draw_box_for_indices(idx_romb, "green")
            draw_wedge_for_indices(idx_wedge, "orange")

        else:
            if key == "hanging_man":
                scatter_dates(idx_hanging, "red", "v", "Висельник")
                draw_vertical_for_indices(idx_hanging, "red")

            elif key == "diamond":
                scatter_dates(idx_diamond, "blue", "D", "Бриллиант")
                draw_box_for_indices(idx_diamond, "blue")

            elif key == "romb":
                scatter_dates(idx_romb, "green", "s", "Ромб")
                draw_box_for_indices(idx_romb, "green")

            elif key == "bearish_harami_cross":
                scatter_dates(idx_harami, "black", "x", "Медвежий крест харами")
                draw_vertical_for_indices(idx_harami, "black")

            elif key == "ascending_wedge":
                scatter_dates(idx_wedge, "orange", "^", "Восходящий клин")
                draw_wedge_for_indices(idx_wedge, "orange")

        if ax_main.get_legend() is None:
            ax_main.legend(loc="best", fontsize=8)

    def highlight_pattern(self, key: Optional[str]):
        """Обработчик пунктов меню: какие паттерны подсвечивать."""
        if key is None:
            self.draw_chart(highlight_key=None)
        else:
            self.draw_chart(highlight_key=key)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
