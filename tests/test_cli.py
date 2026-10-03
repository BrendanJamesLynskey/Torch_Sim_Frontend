from simfront.cli import main


def test_cli_prefill_and_offload(capsys, tmp_path):
    main(["--model", "gpt2", "--tokens", "64", "--save", str(tmp_path / "t.json")])
    out = capsys.readouterr().out
    assert "via dispatch" in out and "H100-SXM" in out and "| `aten.addmm` |" in out
    assert (tmp_path / "t.json").stat().st_size > 0
    main(["--model", "llama3-8b", "--tokens", "64", "--offload", "optical"])
    assert "on the accelerator:" in capsys.readouterr().out


def test_cli_decode_fused_attention(capsys):
    main(["--model", "llama3-8b", "--decode", "256", "--fused"])
    out = capsys.readouterr().out
    assert "'phase': 'decode'" in out and "memory-bound" in out


def test_cli_fake_cpu(capsys):
    main(["--model", "llama3-8b", "--tokens", "128", "--fake-cpu"])
    assert "_scaled_dot_product_flash_attention_for_cpu" in capsys.readouterr().out
