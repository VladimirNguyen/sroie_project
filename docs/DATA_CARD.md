# Thẻ dữ liệu thực nghiệm
Nguồn: https://github.com/metatatt/sorie
Commit: 27be4271b251c256f695acbade9a801bffe85994
Ngày truy cập: 01/11/2026
Bản phân phối: 626 bộ img/box/key; nhãn được nhà phân phối hiệu chỉnh theo README.
Tập con: 300 ảnh, seed42. Train80, val20, test nội bộ200. Không phải split test chính thức.
Nhãn: company/date/address/total, giữ nguyên nội dung; đổi đuôi JSON/CSV sang txt để tương thích loader.
Giấy phép repo: MIT, xem SOURCE_LICENSE.txt. Không khẳng định MIT bao trùm mọi quyền ảnh của bên thứ ba. Trang điều khoản ICDAR không truy cập thành công; xác minh trước khi tái phân phối công khai.
Kiểm tra ảnh trùng byte giữa các tập bằng SHA256; chưa tách theo doanh nghiệp.
Danh sách ID: source_manifest.json và outputs/experiment/splits.json.
