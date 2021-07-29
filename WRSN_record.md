**29/07/2021 - 9AM**: 
1. Thay đổi :
   - hàm reward, nếu hành động đưa ra kết quả là sạc 0s, reward = 0
     + Nếu phạt nặng quá (-5, -10) sẽ làm cho giá trị TD giao động lớn
   - giảm learning rate start xuống vì có hiện tượng hội tụ sớm
     + model trả về PLoss và ELoss là các giá trị nan vì hội tụ sớm (log(0) = nan)
   - sửa truncated_mu trong khoảng từ (0.1,5) thay vì (0,5)
2. Nhận xét:
   - Vẫn có hiện tượng các MC đến cùng một điểm sạc
   - Overlap khá nhiều

**29/07/2021 - 2PM**:
1. Thay đổi:
   - Trong worker, đặt điểm dừng cho việc lấy vector heuristic
   - Đưa hệ số vào H_policy trong hàm *H_get_heuristic_policy*, hiện tại là (0.35, 0.35, 0.4)
2. Nhận xét:
   - Chưa giải quyết được hiện tượng overlap