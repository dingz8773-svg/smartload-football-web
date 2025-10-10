# SmartLoad-Football · 管理后台 (Final Demo v4)

## 本地运行
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 部署到 Streamlit Community Cloud
1. 推送本目录到 GitHub 仓库（任意公开/私有均可）
2. 打开 https://share.streamlit.io → New app
3. 选择你的仓库与分支，App file 选择 `app.py`
4. Deploy 即可获得可分享的公网链接

## 部署到 Hugging Face Spaces
1. 新建 Space（模板选择 **Streamlit**）
2. 上传本目录所有文件，或连接 GitHub 仓库
3. 在设置中将 `app_file` 设为 `app.py`，保存后自动部署

## 中文字体说明
若线上环境图例中文出现方块/乱码，可放置一个中文字体文件：`fonts/NotoSansSC-Regular.otf`，并在 `app.py` 顶部加入：
```python
import matplotlib.font_manager as fm, matplotlib
fm.fontManager.addfont("fonts/NotoSansSC-Regular.otf")
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans SC"]
```
然后重新部署即可。
