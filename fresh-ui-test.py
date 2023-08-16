import gradio as gr

with gr.Blocks(title="RVC WebUI") as app:
    gr.Markdown(
        value=(
            "本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>."
        )
    )
    with gr.Tabs():
        with gr.TabItem(("模型推理")):
            with gr.Row():
                #with gr.Column():
                    sid0 = gr.Dropdown(
                        label=("推理音色"),
                        choices=sorted("a,b,c"),
                    )
                #with gr.Column():
                    refresh_button = gr.Button(
                        ("刷新音色列表和索引路径"),
                        variant="primary",
                    )
                #with gr.Column():
                    clean_button = gr.Button(
                        ("卸载音色省显存"),
                        variant="primary",
                    )
                #with gr.Column():
                    spk_item = gr.Slider(
                        minimum=0,
                        maximum=2333,
                        step=1,
                        label=("请选择说话人id"),
                        value=0,
                        visible=True,   # make it visible always, but
                        interactive=False,  # lock the interactive
                    )
            with gr.Row():
                with gr.Column():
                    with gr.Group() as tab1SingleAudioInput:
                        input_audio0 = gr.Textbox(
                            label="Input Audio file",
                            placeholder="\"D:\\path\\to\\RVC\\todo-songs\\TEST.wav\""
                        )
                        with gr.Accordion( # change extension UI to use accordion (opened by default) for non-invasive usability.
                            label="Learn more",
                            open=False
                        ):
                            gr.Markdown("输入待处理音频文件路径(默认是正确格式示例)")
                    with gr.Group() as tab1SingleIndex:
                        file_index1 = gr.Textbox(
                            label="Select Index (Direct Path)",
                            value="",
                            placeholder="\"D:\\path\\to\\RVC\\logs\\TEST.index\"",
                            interactive=True,
                        )
                        file_index2 = gr.Dropdown(
                            label="Select Index (Dropdown)",
                            choices=sorted("test"),
                            interactive=True,
                        )
                        index_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Index Rate",
                            value=0.75,
                            interactive=True,
                        )
                        with gr.Accordion( # change extension UI to use accordion (opened by default) for non-invasive usability.
                            label="Learn more",
                            open=False
                        ):
                            gr.Markdown("特征检索库文件路径,为空则使用下拉的选择结果")
                            gr.Markdown("自动检测index路径,下拉式选择(dropdown)")
                            gr.Markdown("检索特征占比")

                with gr.Column():
                    with gr.Group() as tab1SingleTransposing:
                        vc_transform0 = gr.Number(
                            label="Transpose",
                            value=0
                        )
                        with gr.Accordion( # change extension UI to use accordion (opened by default) for non-invasive usability.
                            label="Learn more",
                            open=False
                        ):
                            gr.Markdown("变调(整数, 半音数量, 升八度12降八度-12)")
                            gr.Markdown(value=("男转女推荐+12key, 女转男推荐-12key, 如果音域爆炸导致音色失真也可以自己调整到合适音域. "))
                    with gr.Group() as tab1SingleSampling:
                        resample_sr0 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label="Resample",
                            value=0,
                            step=1,
                            interactive=True,
                        )
                        rms_mix_rate0 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Mix Rate",
                            value=0.25,
                            interactive=True,
                        )
                        protect0 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label="Voiceless Protection",
                            value=0.33,
                            step=0.01,
                            interactive=True,
                        )
                        with gr.Accordion( # change extension UI to use accordion (opened by default) for non-invasive usability.
                                label="Learn more",
                                open=False
                            ):
                                gr.Markdown("后处理重采样至最终采样率，0为不进行重采样")
                                gr.Markdown("输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络")
                                gr.Markdown("保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果")

                with gr.Column():
                    with gr.Group() as tab1SingleF0Method:
                        f0method0 = gr.Radio(
                            label="Process Type",
                            choices=["pm", "harvest", "crepe", "rmvpe", "mangio"],
                            value="pm",
                            interactive=True,
                        )
                        with gr.Accordion( # change extension UI to use accordion (opened by default) for non-invasive usability.
                            label="Learn more",
                            open=False
                        ):
                            gr.Markdown("选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU,rmvpe效果最好且微吃GPU")
                    with gr.Group() as tab1SingleFilterRadius:
                            filter_radius0 = gr.Slider(
                                minimum=0,
                                maximum=7,
                                label="Filter Slider",
                                value=3,
                                step=1,
                                interactive=True
                            )
                            with gr.Accordion( # change extension UI to use accordion (opened by default) for non-invasive usability.
                                label="Learn more",
                                open=False
                            ):
                                gr.Markdown("If >=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音")
                    with gr.Accordion(label="F0 file (Optional)", open=False), gr.Group() as tab1SingleOptionalF0File:
                        f0_file = gr.File()
                        with gr.Accordion( # change extension UI to use accordion (opened by default) for non-invasive usability.
                                label="Learn more",
                                open=False
                            ):
                                gr.Markdown("F0曲线文件, 可选, 一行一个音高, 代替默认F0及升降调")

                #with gr.Column():
                with gr.Row() as tab1SingleConvert:
                        but0 = gr.Button(
                            value="Convert",
                            variant="primary",
                            interactive=True
                        )

            with gr.Column(), gr.Group() as tab1SingleAudioOutput:
                vc_output2 = gr.Audio(label=("输出音频(右下角三个点,点了可以下载)"), interactive=False)
                vc_output1 = gr.Textbox(label=("输出信息"), value="Ready", interactive=False)

            with gr.Group(), gr.Accordion(
                label=("批量转换, 输入待转换音频文件夹, 或上传多个音频文件, 在指定文件夹(默认opt)下输出转换的音频. "),
                open=False
                ):
                with gr.Row():
                    with gr.Column():
                        vc_transform1 = gr.Number(
                            label=("变调(整数, 半音数量, 升八度12降八度-12)"), value=0
                        )
                        opt_input = gr.Textbox(label=("指定输出文件夹"), value="opt")
                        f0method1 = gr.Radio(
                            label=(
                                "选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU,rmvpe效果最好且微吃GPU"
                            ),
                            choices=["pm", "harvest", "crepe", "rmvpe"],
                            value="pm",
                            interactive=True,
                        )
                        filter_radius1 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label=(">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音"),
                            value=3,
                            step=1,
                            interactive=True,
                        )
                    with gr.Column():
                        file_index3 = gr.Textbox(
                            label=("特征检索库文件路径,为空则使用下拉的选择结果"),
                            value="",
                            interactive=True,
                        )
                        file_index4 = gr.Dropdown(
                            label=("自动检测index路径,下拉式选择(dropdown)"),
                            choices=sorted("test"),
                            interactive=True,
                        )
                        # file_big_npy2 = gr.Textbox(
                        #     label=("特征文件路径"),
                        #     value="E:\\codes\\py39\\vits_vc_gpu_train\\logs\\mi-test-1key\\total_fea.npy",
                        #     interactive=True,
                        # )
                        index_rate2 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=("检索特征占比"),
                            value=1,
                            interactive=True,
                        )
                    with gr.Column():
                        resample_sr1 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label=("后处理重采样至最终采样率，0为不进行重采样"),
                            value=0,
                            step=1,
                            interactive=True,
                        )
                        rms_mix_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=("输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"),
                            value=1,
                            interactive=True,
                        )
                        protect1 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label=(
                                "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"
                            ),
                            value=0.33,
                            step=0.01,
                            interactive=True,
                        )
                    with gr.Column():
                        dir_input = gr.Textbox(
                            label=("输入待处理音频文件夹路径(去文件管理器地址栏拷就行了)"),
                            value="E:\codes\py39\\test-20230416b\\todo-songs",
                        )
                        inputs = gr.File(
                            file_count="multiple", label=("也可批量拖拽音频文件, 二选一, 优先读文件夹，文件夹留空则读取拖拽文件")
                        )

            with gr.Group(), gr.Accordion(
                    label="Audio Output Format",
                    open=True
                    ):
                    with gr.Row():
                        format1 = gr.Radio(
                            label=("导出文件格式"),
                            choices=["wav", "flac", "mp3", "m4a"],
                            value="flac",
                            interactive=True,
                        )
                        but1 = gr.Button(("转换"), variant="primary")
                    vc_output3 = gr.Textbox(label=("输出信息"), value="Nothing", interactive=False)

        with gr.TabItem(("伴奏人声分离&去混响&去回声")):
            with gr.Group():
                gr.Markdown(
                    value=(
                        "人声伴奏分离批量处理， 使用UVR5模型。 <br>合格的文件夹路径格式举例： E:\\codes\\py39\\vits_vc_gpu\\白鹭霜华测试样例(去文件管理器地址栏拷就行了)。 <br>模型分为三类： <br>1、保留人声：不带和声的音频选这个，对主人声保留比HP5更好。内置HP2和HP3两个模型，HP3可能轻微漏伴奏但对主人声保留比HP2稍微好一丁点； <br>2、仅保留主人声：带和声的音频选这个，对主人声可能有削弱。内置HP5一个模型； <br> 3、去混响、去延迟模型（by FoxJoy）：<br>  (1)MDX-Net(onnx_dereverb):对于双通道混响是最好的选择，不能去除单通道混响；<br>&emsp;(234)DeEcho:去除延迟效果。Aggressive比Normal去除得更彻底，DeReverb额外去除混响，可去除单声道混响，但是对高频重的板式混响去不干净。<br>去混响/去延迟，附：<br>1、DeEcho-DeReverb模型的耗时是另外2个DeEcho模型的接近2倍；<br>2、MDX-Net-Dereverb模型挺慢的；<br>3、个人推荐的最干净的配置是先MDX-Net再DeEcho-Aggressive。"
                    )
                )
                with gr.Row():
                    with gr.Column():
                        dir_wav_input = gr.Textbox(
                            label=("输入待处理音频文件夹路径"),
                            value="E:\\codes\\py39\\test-20230416b\\todo-songs\\todo-songs",
                        )
                        wav_inputs = gr.File(
                            file_count="multiple", label=("也可批量拖拽音频文件, 二选一, 优先读文件夹，文件夹留空则读取拖拽文件")
                        )
                    with gr.Column():
                        model_choose = gr.Dropdown(label=("模型"), choices="uvr5_names")
                        agg = gr.Slider(
                            minimum=0,
                            maximum=20,
                            step=1,
                            label="人声提取激进程度",
                            value=10,
                            interactive=True,
                            visible=False,  # 先不开放调整
                        )
                        opt_vocal_root = gr.Textbox(
                            label=("指定输出主人声文件夹"), value="opt"
                        )
                        opt_ins_root = gr.Textbox(
                            label=("指定输出非主人声文件夹"), value="opt"
                        )
                        format0 = gr.Radio(
                            label=("导出文件格式"),
                            choices=["wav", "flac", "mp3", "m4a"],
                            value="flac",
                            interactive=True,
                        )
                    but2 = gr.Button(("转换"), variant="primary")
                    vc_output4 = gr.Textbox(label=("输出信息"))
        with gr.TabItem(("训练")):
            gr.Markdown(
                value=(
                    "step1: 填写实验配置. 实验数据放在logs下, 每个实验一个文件夹, 需手工输入实验名路径, 内含实验配置, 日志, 训练得到的模型文件. "
                )
            )
            with gr.Row():
                exp_dir1 = gr.Textbox(label=("输入实验名"), value="mi-test")
                sr2 = gr.Radio(
                    label=("目标采样率"),
                    choices=["40k", "48k"],
                    value="40k",
                    interactive=True,
                )
                if_f0_3 = gr.Radio(
                    label=("模型是否带音高指导(唱歌一定要, 语音可以不要)"),
                    choices=[True, False],
                    value=True,
                    interactive=True,
                )
                version19 = gr.Radio(
                    label=("版本"),
                    choices=["v1", "v2"],
                    value="v2",
                    interactive=True,
                    visible=True,
                )
                np7 = gr.Slider(
                    minimum=0,
                    maximum=10,
                    step=1,
                    label=("提取音高和处理数据使用的CPU进程数"),
                    value=int(0),
                    interactive=True,
                )
            with gr.Group():  # 暂时单人的, 后面支持最多4人的#数据处理
                gr.Markdown(
                    value=(
                        "step2a: 自动遍历训练文件夹下所有可解码成音频的文件并进行切片归一化, 在实验目录下生成2个wav文件夹; 暂时只支持单人训练. "
                    )
                )
                with gr.Row():
                    trainset_dir4 = gr.Textbox(
                        label=("输入训练文件夹路径"), value="E:\\语音音频+标注\\米津玄师\\src"
                    )
                    spk_id5 = gr.Slider(
                        minimum=0,
                        maximum=4,
                        step=1,
                        label=("请指定说话人id"),
                        value=0,
                        interactive=True,
                    )
                    but1 = gr.Button(("处理数据"), variant="primary")
                    info1 = gr.Textbox(label=("输出信息"), value="")

            with gr.Group():
                gr.Markdown(value=("step2b: 使用CPU提取音高(如果模型带音高), 使用GPU提取特征(选择卡号)"))
                with gr.Row():
                    with gr.Column():
                        gpus6 = gr.Textbox(
                            label=("以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"),
                            value="gpus",
                            interactive=True,
                            visible=True,
                        )
                        gpu_info9 = gr.Textbox(
                            label=("显卡信息"), value="gpu_info", visible="F0GPUVisible"
                        )
                    with gr.Column():
                        f0method8 = gr.Radio(
                            label=(
                                "选择音高提取算法:输入歌声可用pm提速,高质量语音但CPU差可用dio提速,harvest质量更好但慢,rmvpe效果最好且微吃CPU/GPU"
                            ),
                            choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"],
                            value="rmvpe_gpu",
                            interactive=True,
                        )
                        gpus_rmvpe = gr.Textbox(
                            label=(
                                "rmvpe卡号配置：以-分隔输入使用的不同进程卡号,例如0-0-1使用在卡0上跑2个进程并在卡1上跑1个进程"
                            ),
                            value=4,
                            interactive=True,
                            visible=True,
                        )
                    but2 = gr.Button(("特征提取"), variant="primary")
                    info2 = gr.Textbox(label=("输出信息"), value="", max_lines=8)


            with gr.Group():
                gr.Markdown(value=("step3: 填写训练设置, 开始训练模型和索引"))
                with gr.Row():
                    save_epoch10 = gr.Slider(
                        minimum=1,
                        maximum=50,
                        step=1,
                        label=("保存频率save_every_epoch"),
                        value=5,
                        interactive=True,
                    )
                    total_epoch11 = gr.Slider(
                        minimum=2,
                        maximum=1000,
                        step=1,
                        label=("总训练轮数total_epoch"),
                        value=20,
                        interactive=True,
                    )
                    batch_size12 = gr.Slider(
                        minimum=1,
                        maximum=40,
                        step=1,
                        label=("每张显卡的batch_size"),
                        value="default_batch_size",
                        interactive=True,
                    )
                    if_save_latest13 = gr.Radio(
                        label=("是否仅保存最新的ckpt文件以节省硬盘空间"),
                        choices=[("是"), ("否")],
                        value=("否"),
                        interactive=True,
                    )
                    if_cache_gpu17 = gr.Radio(
                        label=(
                            "是否缓存所有训练集至显存. 10min以下小数据可缓存以加速训练, 大数据缓存会炸显存也加不了多少速"
                        ),
                        choices=[("是"), ("否")],
                        value=("否"),
                        interactive=True,
                    )
                    if_save_every_weights18 = gr.Radio(
                        label=("是否在每次保存时间点将最终小模型保存至weights文件夹"),
                        choices=[("是"), ("否")],
                        value=("否"),
                        interactive=True,
                    )
                with gr.Row():
                    pretrained_G14 = gr.Textbox(
                        label=("加载预训练底模G路径"),
                        value="pretrained_v2/f0G40k.pth",
                        interactive=True,
                    )
                    pretrained_D15 = gr.Textbox(
                        label=("加载预训练底模D路径"),
                        value="pretrained_v2/f0D40k.pth",
                        interactive=True,
                    )
                    gpus16 = gr.Textbox(
                        label=("以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"),
                        value="gpus",
                        interactive=True,
                    )
                    but3 = gr.Button(("训练模型"), variant="primary")
                    but4 = gr.Button(("训练特征索引"), variant="primary")
                    but5 = gr.Button(("一键训练"), variant="primary")
                    info3 = gr.Textbox(label=("输出信息"), value="", max_lines=10)


        with gr.TabItem(("ckpt处理")):
            with gr.Group():
                gr.Markdown(value=("模型融合, 可用于测试音色融合"))
                with gr.Row():
                    ckpt_a = gr.Textbox(label=("A模型路径"), value="", interactive=True)
                    ckpt_b = gr.Textbox(label=("B模型路径"), value="", interactive=True)
                    alpha_a = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=("A模型权重"),
                        value=0.5,
                        interactive=True,
                    )
                with gr.Row():
                    sr_ = gr.Radio(
                        label=("目标采样率"),
                        choices=["40k", "48k"],
                        value="40k",
                        interactive=True,
                    )
                    if_f0_ = gr.Radio(
                        label=("模型是否带音高指导"),
                        choices=[("是"), ("否")],
                        value=("是"),
                        interactive=True,
                    )
                    info__ = gr.Textbox(
                        label=("要置入的模型信息"), value="", max_lines=8, interactive=True
                    )
                    name_to_save0 = gr.Textbox(
                        label=("保存的模型名不带后缀"),
                        value="",
                        max_lines=1,
                        interactive=True,
                    )
                    version_2 = gr.Radio(
                        label=("模型版本型号"),
                        choices=["v1", "v2"],
                        value="v1",
                        interactive=True,
                    )
                with gr.Row():
                    but6 = gr.Button(("融合"), variant="primary")
                    info4 = gr.Textbox(label=("输出信息"), value="", max_lines=8)
            with gr.Group():
                gr.Markdown(value=("修改模型信息(仅支持weights文件夹下提取的小模型文件)"))
                with gr.Row():
                    ckpt_path0 = gr.Textbox(
                        label=("模型路径"), value="", interactive=True
                    )
                    info_ = gr.Textbox(
                        label=("要改的模型信息"), value="", max_lines=8, interactive=True
                    )
                    name_to_save1 = gr.Textbox(
                        label=("保存的文件名, 默认空为和源文件同名"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                with gr.Row():
                    but7 = gr.Button(("修改"), variant="primary")
                    info5 = gr.Textbox(label=("输出信息"), value="", max_lines=8)
            with gr.Group():
                gr.Markdown(value=("查看模型信息(仅支持weights文件夹下提取的小模型文件)"))
                with gr.Row():
                    ckpt_path1 = gr.Textbox(
                        label=("模型路径"), value="", interactive=True
                    )
                    but8 = gr.Button(("查看"), variant="primary")
                    info6 = gr.Textbox(label=("输出信息"), value="", max_lines=8)
            with gr.Group():
                gr.Markdown(
                    value=(
                        "模型提取(输入logs文件夹下大文件模型路径),适用于训一半不想训了模型没有自动提取保存小文件模型,或者想测试中间模型的情况"
                    )
                )
                with gr.Row():
                    ckpt_path2 = gr.Textbox(
                        label=("模型路径"),
                        value="E:\\codes\\py39\\logs\\mi-test_f0_48k\\G_23333.pth",
                        interactive=True,
                    )
                    save_name = gr.Textbox(
                        label=("保存名"), value="", interactive=True
                    )
                    sr__ = gr.Radio(
                        label=("目标采样率"),
                        choices=["32k", "40k", "48k"],
                        value="40k",
                        interactive=True,
                    )
                    if_f0__ = gr.Radio(
                        label=("模型是否带音高指导,1是0否"),
                        choices=["1", "0"],
                        value="1",
                        interactive=True,
                    )
                    version_1 = gr.Radio(
                        label=("模型版本型号"),
                        choices=["v1", "v2"],
                        value="v2",
                        interactive=True,
                    )
                    info___ = gr.Textbox(
                        label=("要置入的模型信息"), value="", max_lines=8, interactive=True
                    )
                    but9 = gr.Button(("提取"), variant="primary")
                    info7 = gr.Textbox(label=("输出信息"), value="", max_lines=8)

        with gr.TabItem(("Onnx导出")):
            with gr.Row():
                ckpt_dir = gr.Textbox(label=("RVC模型路径"), value="", interactive=True)
            with gr.Row():
                onnx_dir = gr.Textbox(
                    label=("Onnx输出路径"), value="", interactive=True
                )
            with gr.Row():
                infoOnnx = gr.Label(label="info")
            with gr.Row():
                butOnnx = gr.Button(("导出Onnx模型"), variant="primary")

        tab_faq = ("常见问题解答")
        with gr.TabItem(tab_faq):
            try:
                with open("docs/faq_en.md", "r", encoding="utf8") as f:
                    info = f.read()
                gr.Markdown(value=info)
            except:
                gr.Markdown(traceback.format_exc())

    app.queue(concurrency_count=511, max_size=1022).launch(
        server_name="0.0.0.0",
        server_port=7898,
        quiet=True,
    )


if __name__ == "__main__":
    app.launch()