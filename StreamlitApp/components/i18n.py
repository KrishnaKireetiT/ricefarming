"""
Internationalization (i18n) module for the Pipeline Testing App.
Supports English (en) and Vietnamese (vi).
"""

import streamlit as st

# ================================================================
# Translation Dictionaries
# ================================================================

TRANSLATIONS = {
    # --- App-level ---
    "app_title": {
        "en": "Pipeline Testing App",
        "vi": "Ứng dụng Đánh giá Pipeline",
    },
    "app_subtitle": {
        "en": "Rice Farming KG Pipeline Evaluation Platform",
        "vi": "Nền tảng Đánh giá Pipeline Đồ thị Tri thức Canh tác Lúa",
    },
    "pipeline_tester": {
        "en": "Pipeline Tester",
        "vi": "Đánh giá Pipeline",
    },

    # --- Login / Signup ---
    "login": {
        "en": "Login",
        "vi": "Đăng nhập",
    },
    "sign_up": {
        "en": "Sign Up",
        "vi": "Đăng ký",
    },
    "username": {
        "en": "Username",
        "vi": "Tên đăng nhập",
    },
    "password": {
        "en": "Password",
        "vi": "Mật khẩu",
    },
    "enter_username": {
        "en": "Enter your username",
        "vi": "Nhập tên đăng nhập",
    },
    "enter_password": {
        "en": "Enter your password",
        "vi": "Nhập mật khẩu",
    },
    "sign_in": {
        "en": "Sign In",
        "vi": "Đăng nhập",
    },
    "choose_username": {
        "en": "Choose a username",
        "vi": "Chọn tên đăng nhập",
    },
    "email_address": {
        "en": "Email address",
        "vi": "Địa chỉ email",
    },
    "choose_password": {
        "en": "Choose a password",
        "vi": "Chọn mật khẩu",
    },
    "min_6_chars": {
        "en": "Min 6 characters",
        "vi": "Tối thiểu 6 ký tự",
    },
    "confirm_password": {
        "en": "Confirm password",
        "vi": "Xác nhận mật khẩu",
    },
    "repeat_password": {
        "en": "Repeat password",
        "vi": "Nhập lại mật khẩu",
    },
    "create_account": {
        "en": "Create Account",
        "vi": "Tạo tài khoản",
    },
    "login_successful": {
        "en": "Login successful!",
        "vi": "Đăng nhập thành công!",
    },
    "invalid_credentials": {
        "en": "Invalid username or password",
        "vi": "Tên đăng nhập hoặc mật khẩu không đúng",
    },
    "fill_all_fields": {
        "en": "Please fill in all fields",
        "vi": "Vui lòng điền tất cả các trường",
    },
    "passwords_no_match": {
        "en": "Passwords do not match",
        "vi": "Mật khẩu không khớp",
    },
    "password_too_short": {
        "en": "Password must be at least 6 characters",
        "vi": "Mật khẩu phải có ít nhất 6 ký tự",
    },
    "username_taken": {
        "en": "Username already taken",
        "vi": "Tên đăng nhập đã được sử dụng",
    },
    "email_registered": {
        "en": "Email already registered",
        "vi": "Email đã được đăng ký",
    },
    "account_created": {
        "en": "Account created! Please login.",
        "vi": "Tạo tài khoản thành công! Vui lòng đăng nhập.",
    },
    "account_failed": {
        "en": "Failed to create account",
        "vi": "Tạo tài khoản thất bại",
    },
    "enter_both": {
        "en": "Please enter both username and password",
        "vi": "Vui lòng nhập tên đăng nhập và mật khẩu",
    },

    # --- Sidebar ---
    "technical_mode": {
        "en": "🔧 Technical Mode",
        "vi": "🔧 Chế độ Kỹ thuật",
    },
    "pipelines": {
        "en": "Pipelines",
        "vi": "Pipelines",
    },
    "both_pipelines_caption": {
        "en": "Both pipelines will run automatically for each question.",
        "vi": "Cả hai pipeline sẽ tự động chạy cho mỗi câu hỏi.",
    },
    "evaluation_progress": {
        "en": "Evaluation Progress",
        "vi": "Tiến độ Đánh giá",
    },
    "questions_evaluated": {
        "en": "questions evaluated",
        "vi": "câu hỏi đã đánh giá",
    },
    "no_evaluations_yet": {
        "en": "No evaluations yet",
        "vi": "Chưa có đánh giá nào",
    },
    "logout": {
        "en": "Logout",
        "vi": "Đăng xuất",
    },
    "pipeline_select": {
        "en": "Pipeline",
        "vi": "Pipeline",
    },
    "check_db": {
        "en": "🔄 Check DB Status",
        "vi": "🔄 Kiểm tra trạng thái CSDL",
    },
    "recent_queries": {
        "en": "📜 Recent Queries",
        "vi": "📜 Truy vấn gần đây",
    },
    "no_queries_yet": {
        "en": "No queries yet",
        "vi": "Chưa có truy vấn nào",
    },
    "language": {
        "en": "Language",
        "vi": "Ngôn ngữ",
    },

    # --- Tab names ---
    "tab_evaluate": {
        "en": "⚖️ Evaluate",
        "vi": "⚖️ Đánh giá",
    },
    "tab_batch_evaluate": {
        "en": "📦 Batch Evaluate",
        "vi": "📦 Đánh giá hàng loạt",
    },
    "tab_results": {
        "en": "📊 Results",
        "vi": "📊 Kết quả",
    },
    "tab_single_query": {
        "en": "🔍 Single Query",
        "vi": "🔍 Truy vấn đơn",
    },
    "tab_batch_testing": {
        "en": "📦 Batch Testing",
        "vi": "📦 Đánh giá hàng loạt",
    },
    "tab_history": {
        "en": "📜 History",
        "vi": "📜 Lịch sử",
    },

    # --- Evaluate tab ---
    "evaluate_pipelines": {
        "en": "Evaluate Pipelines",
        "vi": "Đánh giá Pipeline",
    },
    "evaluate_caption": {
        "en": "Enter a question to see answers from both pipelines side by side. Pipeline identities are hidden until you submit your evaluation.",
        "vi": "Nhập câu hỏi để xem câu trả lời từ hai pipeline cạnh nhau. Danh tính pipeline sẽ ẩn cho đến khi bạn gửi đánh giá.",
    },
    "enter_question": {
        "en": "Enter your question:",
        "vi": "Nhập câu hỏi của bạn:",
    },
    "question_placeholder": {
        "en": "Khi nào nên bón phân đạm cho lúa? / When should I apply nitrogen fertilizer to rice?",
        "vi": "Khi nào nên bón phân đạm cho lúa? / When should I apply nitrogen fertilizer to rice?",
    },
    "run_both_pipelines": {
        "en": "Run Both Pipelines",
        "vi": "Chạy cả hai Pipeline",
    },
    "pipeline_a": {
        "en": "Pipeline A",
        "vi": "Pipeline A",
    },
    "pipeline_b": {
        "en": "Pipeline B",
        "vi": "Pipeline B",
    },
    "pipeline_a_thinking": {
        "en": "Pipeline A thinking...",
        "vi": "Pipeline A đang xử lý...",
    },
    "pipeline_b_thinking": {
        "en": "Pipeline B thinking...",
        "vi": "Pipeline B đang xử lý...",
    },
    "response_time": {
        "en": "Response Time",
        "vi": "Thời gian phản hồi",
    },
    "faster": {
        "en": "Faster",
        "vi": "Nhanh hơn",
    },
    "retrieval_details": {
        "en": "Retrieval details",
        "vi": "Chi tiết truy xuất",
    },

    # --- Evaluation form ---
    "ground_truth_title": {
        "en": "Ground Truth Answer",
        "vi": "Câu trả lời chuẩn",
    },
    "ground_truth_caption": {
        "en": "As a domain expert, write the correct answer to this question. This will be used as a gold standard for automated evaluation later.",
        "vi": "Với tư cách chuyên gia, hãy viết câu trả lời đúng cho câu hỏi này. Câu trả lời này sẽ được dùng làm chuẩn vàng cho đánh giá tự động sau này.",
    },
    "ground_truth_placeholder": {
        "en": "Write what you consider the ideal, complete, and accurate answer to this question...",
        "vi": "Viết câu trả lời bạn cho là lý tưởng, đầy đủ và chính xác cho câu hỏi này...",
    },
    "rate_both_answers": {
        "en": "Rate Both Answers",
        "vi": "Chấm điểm cả hai câu trả lời",
    },
    "score_caption": {
        "en": "Score each pipeline from 1 (poor) to 5 (excellent) on each criterion.",
        "vi": "Chấm điểm mỗi pipeline từ 1 (kém) đến 5 (xuất sắc) theo từng tiêu chí.",
    },
    "overall_preference": {
        "en": "Overall Preference",
        "vi": "Đánh giá tổng thể",
    },
    "pref_a_better": {
        "en": "Pipeline A is better",
        "vi": "Pipeline A tốt hơn",
    },
    "pref_b_better": {
        "en": "Pipeline B is better",
        "vi": "Pipeline B tốt hơn",
    },
    "pref_tie": {
        "en": "Tie (both equal)",
        "vi": "Ngang nhau",
    },
    "additional_notes": {
        "en": "Major Issues & Observations (optional):",
        "vi": "Vấn đề lớn & Nhận xét (không bắt buộc):",
    },
    "notes_placeholder": {
        "en": "Describe any major issues you found in the answers that aren't covered by the tags above. This helps us identify new failure modes we're not aware of...",
        "vi": "Mô tả các vấn đề lớn bạn tìm thấy trong câu trả lời mà chưa được bao gồm trong các thẻ trên. Điều này giúp chúng tôi xác định các loại lỗi mới mà chúng tôi chưa biết...",
    },
    "submit_evaluation": {
        "en": "Submit Evaluation",
        "vi": "Gửi đánh giá",
    },
    "evaluation_saved": {
        "en": "Evaluation saved!",
        "vi": "Đã lưu đánh giá!",
    },
    "evaluation_updated": {
        "en": "Evaluation updated!",
        "vi": "Đã cập nhật đánh giá!",
    },

    # --- Scoring criteria ---
    "criteria_correctness": {
        "en": "Factual Accuracy",
        "vi": "Độ chính xác",
    },
    "criteria_correctness_help": {
        "en": "Are the facts, numbers, and recommendations correct?",
        "vi": "Các thông tin, số liệu và khuyến cáo có chính xác không?",
    },
    "criteria_completeness": {
        "en": "Completeness",
        "vi": "Đầy đủ",
    },
    "criteria_completeness_help": {
        "en": "Does the answer cover all relevant aspects the farmer needs?",
        "vi": "Câu trả lời có bao gồm tất cả các khía cạnh mà nông dân cần không?",
    },
    "criteria_relevance": {
        "en": "Relevance",
        "vi": "Phù hợp",
    },
    "criteria_relevance_help": {
        "en": "Does the answer actually address the question that was asked?",
        "vi": "Câu trả lời có thực sự trả lời đúng câu hỏi được hỏi không?",
    },
    "criteria_specificity": {
        "en": "Specificity",
        "vi": "Cụ thể",
    },
    "criteria_specificity_help": {
        "en": "Does it give concrete details (dosages, timing, varieties) rather than vague advice?",
        "vi": "Có đưa ra chi tiết cụ thể (liều lượng, thời điểm, giống) thay vì lời khuyên chung chung không?",
    },
    "criteria_clarity": {
        "en": "Language & Clarity",
        "vi": "Ngôn ngữ & Rõ ràng",
    },
    "criteria_clarity_help": {
        "en": "Is the answer clear, well-structured, and farmer-friendly?",
        "vi": "Câu trả lời có rõ ràng, có cấu trúc tốt và dễ hiểu với nông dân không?",
    },
    "criteria_harmfulness": {
        "en": "Safety (no harm)",
        "vi": "An toàn (không gây hại)",
    },
    "criteria_harmfulness_help": {
        "en": "Could following this advice cause crop damage or harm? (5 = completely safe, 1 = dangerous)",
        "vi": "Làm theo lời khuyên này có thể gây hại cho cây trồng không? (5 = hoàn toàn an toàn, 1 = nguy hiểm)",
    },

    # --- Failure modes ---
    "failure_modes_title": {
        "en": "Failure Mode Tags",
        "vi": "Loại lỗi",
    },
    "failure_modes_caption": {
        "en": "Select any issues you noticed in each answer. This helps developers identify patterns.",
        "vi": "Chọn các vấn đề bạn nhận thấy trong mỗi câu trả lời. Điều này giúp nhà phát triển nhận diện xu hướng lỗi.",
    },
    "pipeline_a_issues": {
        "en": "Pipeline A issues:",
        "vi": "Vấn đề Pipeline A:",
    },
    "pipeline_b_issues": {
        "en": "Pipeline B issues:",
        "vi": "Vấn đề Pipeline B:",
    },
    "fm_wrong_facts": {
        "en": "Wrong facts",
        "vi": "Sai sự thật",
    },
    "fm_wrong_figures": {
        "en": "Wrong figures",
        "vi": "Số liệu sai",
    },
    "fm_wrong_dates": {
        "en": "Wrong dates",
        "vi": "Ngày tháng sai",
    },
    "fm_missing_info": {
        "en": "Missing information",
        "vi": "Thiếu thông tin",
    },
    "fm_off_topic": {
        "en": "Off-topic response",
        "vi": "Câu trả lời lạc đề",
    },
    "fm_hallucination": {
        "en": "Hallucination (invented information)",
        "vi": "Bịa đặt thông tin",
    },
    "fm_too_vague": {
        "en": "Too vague",
        "vi": "Quá chung chung",
    },
    "fm_outdated": {
        "en": "Outdated recommendation",
        "vi": "Khuyến cáo lỗi thời",
    },
    "fm_harmful": {
        "en": "Unsafe advice",
        "vi": "Lời khuyên không an toàn",
    },

    # --- Results tab ---
    "evaluation_results": {
        "en": "Evaluation Results",
        "vi": "Kết quả Đánh giá",
    },
    "no_evaluations_message": {
        "en": "No evaluations yet. Go to the Evaluate tab to start comparing pipelines!",
        "vi": "Chưa có đánh giá nào. Hãy vào tab Đánh giá để bắt đầu so sánh pipeline!",
    },
    "overview": {
        "en": "Overview",
        "vi": "Tổng quan",
    },
    "evaluated": {
        "en": "Evaluated",
        "vi": "Đã đánh giá",
    },
    "preferred": {
        "en": "Preferred",
        "vi": "Được ưu tiên",
    },
    "tie": {
        "en": "Tie",
        "vi": "Ngang nhau",
    },
    "ground_truths": {
        "en": "Ground Truths",
        "vi": "Câu trả lời chuẩn",
    },
    "avg_scores": {
        "en": "Average Scores (1-5 scale)",
        "vi": "Điểm trung bình (thang 1-5)",
    },
    "criterion": {
        "en": "Criterion",
        "vi": "Tiêu chí",
    },
    "failure_mode_distribution": {
        "en": "Failure Mode Distribution",
        "vi": "Phân bố Loại lỗi",
    },
    "failure_mode_dist_caption": {
        "en": "How often each issue was flagged across all evaluations.",
        "vi": "Tần suất mỗi loại lỗi được ghi nhận qua tất cả đánh giá.",
    },
    "issue": {
        "en": "Issue",
        "vi": "Vấn đề",
    },
    "avg_response_time": {
        "en": "Average Response Time",
        "vi": "Thời gian phản hồi trung bình",
    },
    "evaluation_history": {
        "en": "Evaluation History",
        "vi": "Lịch sử Đánh giá",
    },
    "no_evaluations_found": {
        "en": "No evaluations found.",
        "vi": "Không tìm thấy đánh giá nào.",
    },
    "export_results": {
        "en": "Export Results",
        "vi": "Xuất kết quả",
    },
    "download_csv": {
        "en": "Download CSV",
        "vi": "Tải xuống CSV",
    },
    "download_json": {
        "en": "Download JSON",
        "vi": "Tải xuống JSON",
    },
    "download_excel": {
        "en": "Download Excel",
        "vi": "Tải xuống Excel",
    },

    # --- Batch evaluate ---
    "batch_evaluation": {
        "en": "Batch Evaluation",
        "vi": "Đánh giá hàng loạt",
    },
    "batch_eval_caption": {
        "en": "Upload questions to run both pipelines on each one. Review and score them afterwards.",
        "vi": "Tải lên câu hỏi để chạy cả hai pipeline. Xem và chấm điểm sau đó.",
    },
    "batch_upload_caption": {
        "en": "Upload a list of questions to evaluate both pipelines on each one.",
        "vi": "Tải lên danh sách câu hỏi để đánh giá cả hai pipeline.",
    },
    "run_batch": {
        "en": "Run Batch",
        "vi": "Chạy hàng loạt",
    },
    "batch_completed": {
        "en": "Completed {n} questions! Go to Results to review and score them.",
        "vi": "Hoàn thành {n} câu hỏi! Vào tab Kết quả để xem và chấm điểm.",
    },
    "batch_results": {
        "en": "Batch Results",
        "vi": "Kết quả hàng loạt",
    },
    "questions": {
        "en": "questions",
        "vi": "câu hỏi",
    },
    "input_method": {
        "en": "Input method:",
        "vi": "Phương thức nhập:",
    },
    "upload_file": {
        "en": "Upload file (CSV/TXT)",
        "vi": "Tải tệp lên (CSV/TXT)",
    },
    "paste_questions": {
        "en": "Paste questions",
        "vi": "Dán câu hỏi",
    },
    "upload_help": {
        "en": "Upload questions (CSV with 'question' column, or TXT with one question per line)",
        "vi": "Tải lên câu hỏi (CSV có cột 'question', hoặc TXT mỗi dòng một câu hỏi)",
    },
    "paste_help": {
        "en": "Paste questions (one per line):",
        "vi": "Dán câu hỏi (mỗi dòng một câu):",
    },
    "loaded_n_questions": {
        "en": "Loaded **{n}** questions",
        "vi": "Đã tải **{n}** câu hỏi",
    },

    # --- History / Single Query ---
    "single_query": {
        "en": "Single Query",
        "vi": "Truy vấn đơn",
    },
    "single_query_caption": {
        "en": "Ask a question and inspect the full pipeline response with retrieval details.",
        "vi": "Đặt câu hỏi và xem chi tiết phản hồi pipeline cùng kết quả truy xuất.",
    },
    "run_query": {
        "en": "Run Query",
        "vi": "Chạy truy vấn",
    },
    "running_pipeline": {
        "en": "Running pipeline...",
        "vi": "Đang chạy pipeline...",
    },
    "query_history": {
        "en": "Query History",
        "vi": "Lịch sử truy vấn",
    },
    "back_to_history": {
        "en": "← Back to History",
        "vi": "← Quay lại Lịch sử",
    },
    "entry_not_found": {
        "en": "Entry not found",
        "vi": "Không tìm thấy mục",
    },
    "question_label": {
        "en": "Question",
        "vi": "Câu hỏi",
    },
    "farmer_advisory": {
        "en": "Farmer Advisory",
        "vi": "Tư vấn cho Nông dân",
    },

    # --- Edit evaluation ---
    "edit": {
        "en": "Edit",
        "vi": "Sửa",
    },
    "edit_evaluation": {
        "en": "Edit Evaluation",
        "vi": "Sửa đánh giá",
    },
    "cancel": {
        "en": "Cancel",
        "vi": "Hủy",
    },

    # --- Citations ---
    "citation_chapter": {
        "en": "Chapter",
        "vi": "Chương",
    },
    "citation_section": {
        "en": "Section",
        "vi": "Mục",
    },
    "citation_source_ref": {
        "en": "Source Reference",
        "vi": "Nguồn tham khảo",
    },

    # --- Pipeline loading ---
    "loading_pipeline_a": {
        "en": "Loading Pipeline A...",
        "vi": "Đang tải Pipeline A...",
    },
    "loading_pipeline_b": {
        "en": "Loading Pipeline B...",
        "vi": "Đang tải Pipeline B...",
    },
    "pipeline_a_ready": {
        "en": "✅ Pipeline A ready!",
        "vi": "✅ Pipeline A sẵn sàng!",
    },
    "pipeline_b_ready": {
        "en": "✅ Pipeline B ready!",
        "vi": "✅ Pipeline B sẵn sàng!",
    },
}


# ================================================================
# Helper Functions
# ================================================================

def get_lang() -> str:
    """Get the current UI language from session state."""
    return st.session_state.get("ui_lang", "en")


def t(key: str, **kwargs) -> str:
    """
    Translate a key to the current UI language.

    Args:
        key: Translation key
        **kwargs: Format arguments (e.g., n=5 for "Loaded {n} questions")

    Returns:
        Translated string, or the key itself if not found.
    """
    lang = get_lang()
    entry = TRANSLATIONS.get(key)
    if entry is None:
        return key
    text = entry.get(lang, entry.get("en", key))
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, IndexError):
            pass
    return text


def get_failure_mode_options() -> list:
    """Return failure mode options in the current language."""
    keys = [
        "fm_wrong_facts", "fm_wrong_figures", "fm_wrong_dates",
        "fm_missing_info", "fm_off_topic", "fm_hallucination",
        "fm_too_vague", "fm_outdated", "fm_harmful",
    ]
    return [t(k) for k in keys]


def get_criteria() -> list:
    """Return scoring criteria as list of (label, key, help_text) tuples."""
    return [
        (t("criteria_correctness"), "correctness", t("criteria_correctness_help")),
        (t("criteria_completeness"), "completeness", t("criteria_completeness_help")),
        (t("criteria_relevance"), "relevance", t("criteria_relevance_help")),
        (t("criteria_specificity"), "specificity", t("criteria_specificity_help")),
        (t("criteria_clarity"), "clarity", t("criteria_clarity_help")),
        (t("criteria_harmfulness"), "harmfulness", t("criteria_harmfulness_help")),
    ]
