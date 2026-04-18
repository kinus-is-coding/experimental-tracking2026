
# 🛠 1. Setup ban đầu
Môi trường đã được chuẩn hóa. Anh em cài đặt các thư viện hỗ trợ vẽ plot và quản lý dữ liệu:
    pip install -r requirements.txt
# 🚀 2. Quy trình sau khi Train xong
Sau khi hoàn thành một Run , anh em thực hiện theo 3 bước:
 ## Bước 2.1: Xuất đồ thị chuẩn (Auto-Plot)
 Không tự vẽ bằng Excel hay công cụ khác. Sử dụng module visualizer.py có sẵn trong Repo để xuất hình đúng format Academic
    from scripts.visualizer import save_learning_curves
    # history.csv: file chứa log train_loss, test_loss, train_acc, test_acc qua từng epoch
    save_learning_curves(
        history_csv='path/to/EXPxxx_history.csv', 
        run_id='EXPxxx', 
        output_dir='plot/EXPxxx_RegimeName/'
    )



 ## Bước 2.2: Tổ chức Folder Kết quả
 Mỗi thí nghiệm phải nằm trong folder riêng theo cấu trúc:
plot/[RUN_ID]_[Regime]_[TRAIN/TEST]/

vd: plot/EXP32_REGIME2_TRAIN/

Folder BẮT BUỘC phải có đủ 3 file sau:


history.csv: File log thô từng epoch (để Minh Ngo tổng hợp so sánh).

curves.png: Hình Acc/Loss (tạo từ Bước 2.1, Nhớ đổi tên và kéo vô đúng folder).

writer_acc.csv: Bảng accuracy chi tiết của từng Writer (để tính Writer Bias).
 ## Bước 2.3: Cập nhật Master Tracker
Điền các chỉ số cuối cùng (Final Acc, Final Loss) vào bảng tổng hợp tại đây:
👉 [Link Google Sheets của ông ở đây]