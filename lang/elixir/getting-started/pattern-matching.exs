res = case File.read("/etc/sudoers") do
  {:ok, res} -> res
  {:error, :enoent} -> "oh it isn't here"
  {:error, :eacces} -> "you can't read it" # 管理者権限でないと読めないのでこれが返る。
  _ -> "?"
end

IO.inspect res