try {
    # Git kontrolü
    $gitPath = (Get-Command git -ErrorAction SilentlyContinue).Source
    if (-not $gitPath) {
        Write-Host "❌ Git bulunamadı. Lütfen https://git-scm.com/download/win adresinden yükleyin."
        throw "Git bulunamadı."
    }

    # Proje yolu
    $projectPath = "C:\Users\Alperen\Desktop\cats0dogs-master"
    if (-not (Test-Path $projectPath)) {
        throw "Proje dizini bulunamadı: $projectPath"
    }

    Set-Location $projectPath
    Write-Host "📂 Proje dizinine geçildi: $projectPath"

    # Commit mesajı al
    $commitMessage = Read-Host "📝 Commit mesajını girin"

    # Git işlemleri
    git add .
    git commit -m "$commitMessage"
    $branch = (git rev-parse --abbrev-ref HEAD).Trim()
    if (-not $branch) { $branch = "main" }

    Write-Host "🚀 GitHub'a gönderiliyor..."
    git push origin $branch

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Güncelleme başarıyla tamamlandı ($branch dalı)."
    } else {
        Write-Host "⚠️ Push işlemi sırasında hata oluştu."
    }
}
catch {
    Write-Host "⚠️ Hata: $_"
}
finally {
    Write-Host "`nİşlem tamamlandı. Devam etmek için bir tuşa basın..."
    Pause
}
