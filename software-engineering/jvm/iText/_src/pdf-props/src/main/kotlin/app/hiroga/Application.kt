package app.hiroga

import com.itextpdf.kernel.pdf.PdfDocument
import com.itextpdf.kernel.pdf.PdfReader

fun main(args: Array<String>) {
    args.forEach { readPdf(it) }
}

fun readPdf(fileName: String) {
    val reader = PdfReader(fileName)
    val pdfDoc = PdfDocument(reader)
    val pdfObject = pdfDoc.catalog.viewerPreferences.pdfObject
    println(pdfObject)
}
