import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from geneticalgorithm import geneticalgorithm as ga
import joblib
import sqlite3
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from datetime import datetime, timedelta
import base64
import warnings
warnings.filterwarnings('ignore')

# ---------------------- تنظیمات اولیه پیشرفته ----------------------
st.set_page_config(
    page_title="مشاور هوشمند کنکور PRO",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تنظیم تم دارک و فونت
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("./font/fonts.css")

# ---------------------- اتصال به دیتابیس SQLite پیشرفته ----------------------
def init_db():
    conn = sqlite3.connect('student_data_pro.db', check_same_thread=False)
    c = conn.cursor()
    
    # ایجاد جداول پیشرفته
    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            student_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            field TEXT,
            target_major TEXT,
            register_date DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            lesson TEXT,
            topic TEXT,
            score INTEGER,
            difficulty INTEGER,
            study_time FLOAT,
            error_type TEXT,
            test_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students (student_id)
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS training_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            model_type TEXT,
            accuracy FLOAT,
            f1_score FLOAT,
            roc_auc FLOAT,
            features TEXT,
            parameters TEXT
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS study_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            plan_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            total_hours FLOAT,
            plan_details TEXT,
            FOREIGN KEY (student_id) REFERENCES students (student_id)
        )
    ''')

    conn.commit()
    return conn

conn = init_db()

# ---------------------- توابع کمکی پیشرفته ----------------------
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        background-attachment: fixed;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# تنظیم پس‌زمینه (اختیاری)
# set_background('background.png')

# ---------------------- تولید داده‌های نمونه هوشمند ----------------------
def generate_smart_sample_data(student_id=1):
    lessons = {
        "ریاضی": ["تابع", "مثلثات", "حد و پیوستگی", "مشتق", "انتگرال"],
        "فیزیک": ["سینماتیک", "دینامیک", "نوسان", "الکتریسیته", "مغناطیس"],
        "شیمی": ["استوکیومتری", "ترمودینامیک", "سینتیک", "تعادل", "اسید و باز"],
        "ادبیات": ["املا", "تاریخ ادبیات", "آرایه", "قرابت", "زبان فارسی"],
        "زبان": ["گرامر", "واژگان", "درک مطلب", "کلوز تست", "تلفظ"]
    }
    
    data = []
    for lesson, topics in lessons.items():
        for topic in topics:
            difficulty = np.random.randint(1, 6)
            base_score = np.random.randint(10, 20)
            score = max(0, min(20, base_score + np.random.randint(-3, 4) - (difficulty-3)))
            study_time = np.random.uniform(2.0, 8.0)
            error_types = ["مفهومی", "محاسباتی", "تستی", "تمرینی", "زمان"]
            error_type = np.random.choice(error_types, p=[0.4, 0.3, 0.15, 0.1, 0.05])
            
            data.append({
                "student_id": student_id,
                "lesson": lesson,
                "topic": topic,
                "score": score,
                "difficulty": difficulty,
                "study_time": round(study_time, 1),
                "error_type": error_type,
                "test_date": (datetime.now() - timedelta(days=np.random.randint(0, 30))).strftime("%Y-%m-%d")
            })
    
    return pd.DataFrame(data)

# ---------------------- مدل‌های پیشرفته یادگیری ماشین ----------------------
def train_advanced_model(data, threshold=15, model_type='ensemble'):
    # پیش‌پردازش داده‌ها
    le = LabelEncoder()
    if os.path.exists('models/label_encoder_pro.pkl'):
        le = joblib.load('models/label_encoder_pro.pkl')
    
    data['lesson_code'] = le.fit_transform(data['lesson'])
    os.makedirs('models', exist_ok=True)
    joblib.dump(le, 'models/label_encoder_pro.pkl')
    
    # نرمال‌سازی داده‌ها
    scaler = MinMaxScaler()
    features = data[['score', 'difficulty', 'lesson_code', 'study_time']]
    scaled_features = scaler.fit_transform(features)
    joblib.dump(scaler, 'models/scaler_pro.pkl')
    
    # تعریف هدف
    data['ضعف'] = (data['score'] < threshold).astype(int)
    
    # تقسیم داده‌ها
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_features, data['ضعف'], test_size=0.2, stratify=data['ضعف'], random_state=42
    )
    
    # آموزش مدل انتخابی
    if model_type == 'xgboost':
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42
        )
    elif model_type == 'lgbm':
        model = LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42
        )
    elif model_type == 'catboost':
        model = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3,
            random_strength=1,
            border_count=32,
            verbose=0,
            random_state=42
        )
    else:  # ensemble
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        lgbm = LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        cat = CatBoostClassifier(
            iterations=200,
            depth=5,
            learning_rate=0.1,
            verbose=0,
            random_state=42
        )
        model = VotingClassifier(
            estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat)],
            voting='soft'
        )
    
    # آموزش با اعتبارسنجی متقابل
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        cv_scores.append(f1_score(y_val_fold, y_pred))
    
    # آموزش نهایی
    model.fit(X_train, y_train)
    
    # ارزیابی مدل
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = model.score(X_test, y_test)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred)
    
    # ذخیره مدل
    model_file = f'models/weakness_model_{model_type}_pro.pkl'
    joblib.dump(model, model_file)
    joblib.dump(features.columns.tolist(), 'models/feature_names_pro.pkl')
    
    # ذخیره تاریخچه آموزش
    conn.execute('''
        INSERT INTO training_history 
        (model_type, accuracy, f1_score, roc_auc, features, parameters)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        model_type,
        accuracy,
        f1,
        roc_auc,
        str(features.columns.tolist()),
        str(model.get_params())
    ))
    conn.commit()
    
    return model, report, accuracy, f1, roc_auc

# ---------------------- بهینه‌سازی برنامه مطالعاتی پیشرفته ----------------------
def advanced_study_plan_optimization(weakness_scores, difficulties, total_hours=20, student_id=None):
    # محاسبه اولویت‌ها
    priorities = weakness_scores * difficulties
    
    # تابع هدف پیشرفته
    def objective(x):
        # بخش اصلی: بیشینه‌سازی تاثیر مطالعه
        main_score = -np.sum(priorities * x)
        
        # جریمه برای انحراف از زمان کل
        penalty = 1000 * abs(np.sum(x) - total_hours)
        
        # جریمه برای توزیع نامتوازن
        balance_penalty = 500 * np.std(x)
        
        # پاداش برای تخصیص زمان به نقاط ضعف
        weakness_bonus = -200 * np.sum(weakness_scores * x)
        
        return main_score + penalty + balance_penalty + weakness_bonus
    
    # محدودیت‌ها
    varbounds = np.array([[0.5, total_hours]] * len(weakness_scores))
    
    # پارامترهای الگوریتم ژنتیک پیشرفته
    algorithm_param = {
        'max_num_iteration': 300,
        'population_size': 150,
        'mutation_probability': 0.15,
        'elit_ratio': 0.15,
        'crossover_probability': 0.7,
        'crossover_type': 'two_point',
        'parents_portion': 0.4,
        'max_iteration_without_improv': 50
    }
    
    # اجرای الگوریتم
    model = ga(
        function=objective,
        dimension=len(weakness_scores),
        variable_type='real',
        variable_boundaries=varbounds,
        algorithm_parameters=algorithm_param
    )
    
    model.run()
    
    # نرمال‌سازی نتایج
    optimized_hours = model.output_dict['variable']
    optimized_hours = optimized_hours * (total_hours / np.sum(optimized_hours))
    
    # ذخیره برنامه در دیتابیس
    if student_id:
        plan_details = {
            'topics': weakness_scores.index.tolist(),
            'hours': np.round(optimized_hours, 1).tolist(),
            'priorities': priorities.tolist()
        }
        
        conn.execute('''
            INSERT INTO study_plans (student_id, total_hours, plan_details)
            VALUES (?, ?, ?)
        ''', (student_id, total_hours, str(plan_details)))
        conn.commit()
    
    return np.round(optimized_hours, 1)

# ---------------------- تحلیل پیشرفته نقاط ضعف ----------------------
def advanced_weakness_analysis(df, model):
    try:
        # بارگذاری پیش‌پردازش‌گرها
        le = joblib.load('models/label_encoder_pro.pkl')
        scaler = joblib.load('models/scaler_pro.pkl')
        feature_names = joblib.load('models/feature_names_pro.pkl')
        
        # پیش‌پردازش داده‌ها
        df['lesson_code'] = le.transform(df['lesson'])
        features = df[feature_names]
        scaled_features = scaler.transform(features)
        
        # پیش‌بینی
        df['ضعف'] = model.predict(scaled_features)
        df['احتمال ضعف'] = model.predict_proba(scaled_features)[:, 1]
        
        # محاسبه شاخص‌های تحلیلی
        df['اولویت'] = df['ضعف'] * df['difficulty'] * (1 + df['احتمال ضعف'])
        df['بازدهی'] = df['score'] / df['study_time']
        df['نیاز به بهبود'] = (20 - df['score']) * df['difficulty']
        
        return df.sort_values('اولویت', ascending=False)
    
    except Exception as e:
        st.error(f"خطا در تحلیل داده‌ها: {str(e)}")
        return df

# ---------------------- رابط کاربری حرفه‌ای ----------------------
def main():
    # نوار کناری - مدیریت دانش‌آموزان
    st.sidebar.header("👨‍🎓 مدیریت دانش‌آموزان")
    with st.sidebar.expander("➕ دانش‌آموز جدید"):
        with st.form("student_form"):
            name = st.text_input("نام کامل")
            field = st.selectbox("رشته", ["ریاضی", "تجربی", "انسانی", "هنر", "زبان"])
            target_major = st.text_input("هدف تحصیلی (رشته و دانشگاه)")
            
            if st.form_submit_button("ثبت دانش‌آموز"):
                conn.execute('''
                    INSERT INTO students (name, field, target_major) 
                    VALUES (?, ?, ?)
                ''', (name, field, target_major))
                conn.commit()
                st.success("دانش‌آموز با موفقیت ثبت شد!")
    
    # انتخاب دانش‌آموز
    students_df = pd.read_sql("SELECT * FROM students", conn)
    if not students_df.empty:
        student_id = st.sidebar.selectbox(
            "انتخاب دانش‌آموز",
            students_df['student_id'],
            format_func=lambda x: f"{students_df[students_df['student_id']==x]['name'].iloc[0]} (ID: {x})"
        )
    else:
        student_id = None
        st.sidebar.warning("ابتدا دانش‌آموز جدید ثبت کنید")
    
    # نوار کناری - ورود داده‌های آزمون
    st.sidebar.header("📝 ورود داده‌های آزمون")
    with st.sidebar.form("score_form", clear_on_submit=True):
        lesson = st.selectbox("درس", ["ریاضی", "فیزیک", "شیمی", "ادبیات", "زبان", "دینی", "عربی"])
        topic = st.text_input("مبحث")
        score = st.slider("نمره", 0, 20, 12)
        difficulty = st.slider("سختی مبحث", 1, 5, 3)
        study_time = st.number_input("زمان مطالعه هفتگی (ساعت)", 0.0, 30.0, 4.0, 0.5)
        error_type = st.selectbox("نوع خطا", ["مفهومی", "محاسباتی", "تستی", "تمرینی", "زمان", "بی‌دقتی"])
        test_date = st.date_input("تاریخ آزمون", datetime.now())
        
        if st.form_submit_button("ذخیره نمره"):
            if student_id:
                conn.execute('''
                    INSERT INTO scores 
                    (student_id, lesson, topic, score, difficulty, study_time, error_type, test_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (student_id, lesson, topic, score, difficulty, study_time, error_type, test_date.strftime("%Y-%m-%d")))
                conn.commit()
                st.success("✅ داده‌ها با موفقیت ذخیره شدند!")
            else:
                st.error("لطفاً ابتدا دانش‌آموز را انتخاب کنید")
    
    # نوار کناری - مدیریت مدل
    st.sidebar.header("🤖 مدیریت مدل هوش مصنوعی")
    model_type = st.sidebar.selectbox(
        "نوع مدل", 
        ['ensemble', 'xgboost', 'lgbm', 'catboost'],
        help="مدل Ensemble ترکیبی از تمام مدل‌ها با دقت بالاتر است"
    )
    threshold = st.sidebar.slider("آستانه تشخیص ضعف", 0, 20, 15, 1)
    
    if st.sidebar.button("آموزش مدل جدید", help="مدل جدید را بر اساس داده‌های فعلی آموزش می‌دهد"):
        if student_id:
            df = pd.read_sql(f"SELECT * FROM scores WHERE student_id={student_id}", conn)
            if len(df) > 10:
                with st.spinner("در حال آموزش مدل پیشرفته..."):
                    try:
                        model, report, accuracy, f1, roc_auc = train_advanced_model(df, threshold, model_type)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("دقت مدل", f"{accuracy:.2%}")
                        col2.metric("F1 Score", f"{f1:.2f}")
                        col3.metric("ROC AUC", f"{roc_auc:.2f}")
                        
                        with st.expander("جزئیات گزارش طبقه‌بندی"):
                            st.code(report)
                    except Exception as e:
                        st.error(f"خطا در آموزش مدل: {str(e)}")
            else:
                st.warning("حداقل به 10 نمونه داده برای آموزش مدل نیاز است")
        else:
            st.error("لطفاً ابتدا دانش‌آموز را انتخاب کنید")
    
    # محتوای اصلی
    st.title("🎯 مشاور هوشمند کنکور PRO")
    st.markdown("""
        <style>
        .big-font {
            font-size:18px !important;
            color: #f0f2f6;
        }
        </style>
        <p class="big-font">سیستم هوشمند تحلیل نقاط ضعف و برنامه‌ریزی مطالعاتی پیشرفته</p>
    """, unsafe_allow_html=True)
    
    if student_id:
        student_info = students_df[students_df['student_id']==student_id].iloc[0]
        
        # نمایش اطلاعات دانش‌آموز
        with st.container():
            col1, col2, col3 = st.columns(3)
            col1.metric("نام دانش‌آموز", student_info['name'])
            col2.metric("رشته", student_info['field'])
            col3.metric("هدف تحصیلی", student_info['target_major'])
        
        # بارگذاری داده‌ها
        df = pd.read_sql(f"SELECT * FROM scores WHERE student_id={student_id}", conn)
        
        if not df.empty:
            # تب‌های تحلیل
            tab1, tab2, tab3, tab4 = st.tabs(["📊 تحلیل کلی", "📚 برنامه مطالعاتی", "📈 پیشرفت تحصیلی", "⚙️ تنظیمات پیشرفته"])
            
            with tab1:
                st.subheader("تحلیل جامع عملکرد تحصیلی")
                
                try:
                    # بارگذاری مدل
                    model_file = f'models/weakness_model_{model_type}_pro.pkl'
                    if os.path.exists(model_file):
                        model = joblib.load(model_file)
                        df_analyzed = advanced_weakness_analysis(df.copy(), model)
                        
                        # نمایش بصری
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig = px.pie(
                                df_analyzed, 
                                names='lesson', 
                                values='score',
                                title='توزیع نمرات بر اساس درس',
                                hole=0.3,
                                color_discrete_sequence=px.colors.sequential.RdBu
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            fig = px.bar(
                                df_analyzed.sort_values('بازدهی', ascending=False),
                                x='topic',
                                y='بازدهی',
                                color='lesson',
                                title='بازدهی مطالعاتی (نمره به ازای هر ساعت مطالعه)',
                                labels={'بازدهی': 'بازدهی (نمره/ساعت)'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            fig = px.treemap(
                                df_analyzed,
                                path=['lesson', 'topic'],
                                values='نیاز به بهبود',
                                color='احتمال ضعف',
                                color_continuous_scale='RdYlGn_r',
                                title='نقاط ضعف بر اساس درس و مبحث'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            fig = px.scatter(
                                df_analyzed,
                                x='difficulty',
                                y='score',
                                color='lesson',
                                size='study_time',
                                hover_name='topic',
                                title='رابطه سختی مبحث با نمره و زمان مطالعه',
                                labels={'difficulty': 'سختی مبحث', 'score': 'نمره'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # جدول تحلیلی
                        st.subheader("جدول تحلیلی نقاط ضعف و قوت")
                        st.dataframe(
                            df_analyzed[['lesson', 'topic', 'score', 'difficulty', 'study_time', 
                                        'error_type', 'ضعف', 'احتمال ضعف', 'اولویت']]
                            .sort_values('اولویت', ascending=False)
                            .style.background_gradient(cmap='Reds', subset=['احتمال ضعف', 'اولویت'])
                            .format({'احتمال ضعف': "{:.2%}"}),
                            use_container_width=True
                        )
                    
                    else:
                        st.warning("مدل آموزش دیده‌ای یافت نشد. لطفاً ابتدا مدل را آموزش دهید.")
                
                except Exception as e:
                    st.error(f"خطا در تحلیل داده‌ها: {str(e)}")
            
            with tab2:
                st.subheader("برنامه‌ریزی هوشمند مطالعاتی")
                
                if os.path.exists(model_file):
                    total_hours = st.slider("⏳ کل ساعات مطالعه هفتگی:", 10, 40, 20, 1,
                                          help="تعداد ساعاتی که می‌توانید در هفته مطالعه کنید")
                    
                    if st.button("🔄 تولید برنامه بهینه", key="optimize_btn"):
                        with st.spinner("در حال بهینه‌سازی برنامه مطالعاتی..."):
                            try:
                                df_analyzed = advanced_weakness_analysis(df.copy(), model)
                                weakness_scores = df_analyzed.set_index('topic')['احتمال ضعف']
                                difficulties = df_analyzed.set_index('topic')['difficulty']
                                
                                optimized_hours = advanced_study_plan_optimization(
                                    weakness_scores, 
                                    difficulties, 
                                    total_hours,
                                    student_id
                                )
                                
                                df_analyzed['زمان بهینه'] = optimized_hours
                                df_analyzed = df_analyzed.sort_values('اولویت', ascending=False)
                                
                                # نمایش برنامه
                                st.success("🎯 برنامه بهینه‌سازی شده:")
                                
                                # نمایش گرافیکی
                                fig = px.bar(
                                    df_analyzed.head(10),
                                    x='topic',
                                    y='زمان بهینه',
                                    color='lesson',
                                    title='توزیع زمان مطالعه بهینه بر اساس مباحث',
                                    labels={'زمان بهینه': 'ساعات مطالعه هفتگی'}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # نمایش جزئیات برنامه
                                for idx, row in df_analyzed.iterrows():
                                    if row['زمان بهینه'] > 0.5:
                                        with st.expander(f"{row['topic']} ({row['lesson']}) - {row['زمان بهینه']} ساعت"):
                                            col1, col2 = st.columns(2)
                                            col1.metric("اولویت", f"{row['اولویت']:.1f}")
                                            col2.metric("احتمال ضعف", f"{row['احتمال ضعف']:.0%}")
                                            
                                            st.progress(row['احتمال ضعف'])
                                            
                                            if row['error_type'] == "مفهومی":
                                                st.info("""
                                                **راهکار:**  
                                                - مشاهده فیلم آموزشی مفهومی  
                                                - مطالعه کتاب درسی پایه  
                                                - حل مثال‌های تشریحی  
                                                """)
                                            elif row['error_type'] == "محاسباتی":
                                                st.info("""
                                                **راهکار:**  
                                                - تمرین روزانه محاسبات سریع  
                                                - استفاده از تکنیک‌های تستی  
                                                - زمان‌بندی حل مسئله  
                                                """)
                                            else:
                                                st.info("""
                                                **راهکار:**  
                                                - حل تست‌های زمان‌دار  
                                                - بررسی خطاهای متداول  
                                                - مرور نکات کلیدی  
                                                """)
                                            
                                            st.markdown(f"""
                                            **منابع پیشنهادی:**  
                                            - [فیلم آموزشی {row['topic']}](https://example.com)  
                                            - کتاب تست {row['lesson']} انتشارات خیلی سبز  
                                            - بانک سوالات کنکور ۱۴۰۱-۱۴۰۲  
                                            """)
                                
                                # ذخیره برنامه
                                csv = df_analyzed[['lesson', 'topic', 'زمان بهینه', 'اولویت']].to_csv(index=False)
                                st.download_button(
                                    label="📥 دانلود برنامه مطالعاتی",
                                    data=csv,
                                    file_name=f'study_plan_{student_id}.csv',
                                    mime='text/csv'
                                )
                            
                            except Exception as e:
                                st.error(f"خطا در بهینه‌سازی: {str(e)}")
                else:
                    st.warning("لطفاً ابتدا مدل را آموزش دهید")
            
            with tab3:
                st.subheader("پیگیری پیشرفت تحصیلی")
                
                # تحلیل زمانی
                df['test_date'] = pd.to_datetime(df['test_date'])
                df = df.sort_values('test_date')
                
                # محاسبه میانگین متحرک
                df['moving_avg'] = df['score'].rolling(window=3, min_periods=1).mean()
                
                fig = px.line(
                    df,
                    x='test_date',
                    y=['score', 'moving_avg'],
                    color='lesson',
                    title='روند تغییرات نمرات در طول زمان',
                    labels={'value': 'نمره', 'test_date': 'تاریخ آزمون'},
                    hover_data=['topic', 'difficulty']
                )
                fig.update_traces(mode='markers+lines')
                st.plotly_chart(fig, use_container_width=True)
                
                # مقایسه درس‌ها
                fig = px.box(
                    df,
                    x='lesson',
                    y='score',
                    color='lesson',
                    title='توزیع نمرات در درس‌های مختلف',
                    labels={'score': 'نمره', 'lesson': 'درس'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # تحلیل خطاها
                fig = px.sunburst(
                    df,
                    path=['lesson', 'error_type'],
                    values='study_time',
                    title='توزیع زمان مطالعه بر اساس نوع خطا'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.subheader("تنظیمات پیشرفته")
                
                with st.expander("مدیریت داده‌ها"):
                    st.dataframe(df, use_container_width=True)
                    
                    if st.button("بارگذاری داده‌های نمونه"):
                        sample_df = generate_smart_sample_data(student_id)
                        sample_df.to_sql('scores', conn, if_exists='append', index=False)
                        st.success("داده‌های نمونه با موفقیت بارگذاری شدند!")
                        st.rerun()
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="دانلود داده‌ها (CSV)",
                        data=csv,
                        file_name='student_scores.csv',
                        mime='text/csv'
                    )
                
                with st.expander("تاریخچه آموزش مدل"):
                    history_df = pd.read_sql(
                        "SELECT * FROM training_history ORDER BY timestamp DESC", 
                        conn
                    )
                    st.dataframe(history_df, use_container_width=True)
                
                with st.expander("برنامه‌های مطالعاتی قبلی"):
                    plans_df = pd.read_sql(
                        f"SELECT * FROM study_plans WHERE student_id={student_id} ORDER BY plan_date DESC", 
                        conn
                    )
                    
                    if not plans_df.empty:
                        for idx, row in plans_df.iterrows():
                            with st.expander(f"برنامه {row['plan_date']} - {row['total_hours']} ساعت"):
                                plan_details = eval(row['plan_details'])
                                fig = px.pie(
                                    names=plan_details['topics'],
                                    values=plan_details['hours'],
                                    title='توزیع ساعات مطالعه'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("برنامه مطالعاتی ثبت نشده است")
        
        else:
            st.warning("هنوز داده‌ای برای این دانش‌آموز ثبت نشده است")
            
            if st.button("بارگذاری داده‌های نمونه"):
                sample_df = generate_smart_sample_data(student_id)
                sample_df.to_sql('scores', conn, if_exists='append', index=False)
                st.success("داده‌های نمونه با موفقیت بارگذاری شدند!")
                st.rerun()
    
    else:
        st.warning("لطفاً از نوار کناری یک دانش‌آموز انتخاب کنید")

if __name__ == "__main__":
    main()