//! Local document text extraction: PDF (text layer), DOCX, and XLSX/XLS/ODS.
//!
//! Text is pulled out on the machine, so any text-only model can read these files
//! offline, without a multimodal cloud model.
//!
//! No OCR: a scanned (image-only) PDF has no text layer, so extraction yields nothing.
//! That case is reported via `Extracted::scanned_pdf` so the caller can route it to a
//! vision model instead (text PDF and image PDF are different inputs, not an error).

use std::io::Read;
use std::path::Path;

/// Document kinds we extract text from locally, keyed by file extension.
#[derive(Clone, Copy)]
pub enum DocKind {
    Pdf,
    Docx,
    Xlsx,
}

impl DocKind {
    /// Classify a path by extension. `None` → not a document we extract (treat as text).
    pub fn from_path(path: &Path) -> Option<DocKind> {
        let ext = path.extension()?.to_str()?.to_ascii_lowercase();
        match ext.as_str() {
            "pdf" => Some(DocKind::Pdf),
            "docx" => Some(DocKind::Docx),
            "xlsx" | "xlsm" | "xls" | "ods" => Some(DocKind::Xlsx),
            _ => None,
        }
    }
}

/// The result of extracting a document.
pub struct Extracted {
    /// The extracted UTF-8 text (empty only for a genuinely empty document, or a scanned
    /// PDF where `scanned_pdf` is set).
    pub text: String,
    /// True only for a PDF whose text layer is empty - i.e. a scanned/image-only PDF that
    /// needs OCR or a vision model. Lets the caller route to vision instead of returning
    /// nothing.
    pub scanned_pdf: bool,
}

/// Extract text from a document. `Err(msg)` is a REAL failure (corrupt/unreadable/not the
/// claimed format) surfaced loudly - never a silently-swallowed one.
pub fn extract(path: &Path, kind: DocKind) -> Result<Extracted, String> {
    match kind {
        DocKind::Pdf => extract_pdf(path),
        DocKind::Docx => Ok(Extracted {
            text: extract_docx(path)?,
            scanned_pdf: false,
        }),
        DocKind::Xlsx => Ok(Extracted {
            text: extract_xlsx(path)?,
            scanned_pdf: false,
        }),
    }
}

/// PDF text-layer extraction. An empty result means no text layer (scanned) - flagged,
/// not failed.
fn extract_pdf(path: &Path) -> Result<Extracted, String> {
    let text = pdf_extract::extract_text(path).map_err(|e| format!("pdf: {e}"))?;
    if text.trim().is_empty() {
        return Ok(Extracted {
            text: String::new(),
            scanned_pdf: true,
        });
    }
    Ok(Extracted {
        text,
        scanned_pdf: false,
    })
}

/// DOCX text extraction. A .docx is a zip; the body text lives in `word/document.xml`
/// inside `<w:t>` runs, with paragraphs delimited by `</w:p>`. We read that one entry and
/// pull the text out - no full XML DOM, no OCR (a Word doc always has a text layer).
fn extract_docx(path: &Path) -> Result<String, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("docx: open: {e}"))?;
    let mut zip = zip::ZipArchive::new(file).map_err(|e| format!("docx: not a zip: {e}"))?;
    let mut xml = String::new();
    zip.by_name("word/document.xml")
        .map_err(|e| format!("docx: no word/document.xml (not a Word file?): {e}"))?
        .read_to_string(&mut xml)
        .map_err(|e| format!("docx: read body: {e}"))?;
    Ok(strip_docx_xml(&xml))
}

/// Pull text out of WordprocessingML: content inside `<w:t>…</w:t>` is literal text;
/// `</w:p>` ends a paragraph (newline); `<w:tab/>` and `<w:br/>` map to tab/newline.
fn strip_docx_xml(xml: &str) -> String {
    let bytes = xml.as_bytes();
    let mut out = String::new();
    let mut in_text = false; // inside a <w:t> run
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'<' {
            let start = i;
            while i < bytes.len() && bytes[i] != b'>' {
                i += 1;
            }
            let end = (i + 1).min(xml.len());
            let tag = &xml[start..end];
            i += 1; // step past '>'
            if tag.starts_with("<w:t>") || tag.starts_with("<w:t ") {
                in_text = true;
            } else if tag.starts_with("</w:t>") {
                in_text = false;
            } else if tag.starts_with("</w:p>") {
                out.push('\n');
            } else if tag.starts_with("<w:tab") {
                out.push('\t');
            } else if tag.starts_with("<w:br") {
                out.push('\n');
            }
        } else {
            let start = i;
            while i < bytes.len() && bytes[i] != b'<' {
                i += 1;
            }
            if in_text {
                out.push_str(&xml[start..i]);
            }
        }
    }
    unescape_xml(out.trim_end())
}

/// Unescape the five XML predefined entities. `&amp;` last so a literal `&lt;` in the
/// source never gets double-decoded.
fn unescape_xml(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

/// Extract ONE named sheet as tab-separated rows - for targeted reads of a big workbook
/// after the AI has seen the manifest and chosen a sheet, instead of dumping all sheets.
pub fn extract_sheet(path: &Path, sheet: &str) -> Result<String, String> {
    use calamine::{open_workbook_auto, Reader};
    let mut wb = open_workbook_auto(path).map_err(|e| format!("xlsx: open: {e}"))?;
    // Accept a 1-based INDEX as well as a name, since names may be empty/meaningless.
    let names = wb.sheet_names().to_owned();
    let name = match sheet.parse::<usize>() {
        Ok(n) => names
            .get(n.wrapping_sub(1))
            .cloned()
            .ok_or_else(|| format!("xlsx: sheet index {n} out of range (1..={})", names.len()))?,
        Err(_) => sheet.to_string(),
    };
    let range = wb
        .worksheet_range(&name)
        .map_err(|e| format!("xlsx: sheet '{name}': {e}"))?;
    let mut out = String::new();
    for row in range.rows() {
        let cells: Vec<String> = row.iter().map(|c| c.to_string()).collect();
        out.push_str(&cells.join("\t"));
        out.push('\n');
    }
    Ok(out.trim_end().to_string())
}

/// XLSX/XLS/ODS extraction via calamine. Every sheet is rendered as tab-separated rows
/// under a `# Sheet: <name>` header - a compact, model-readable table dump.
fn extract_xlsx(path: &Path) -> Result<String, String> {
    use calamine::{open_workbook_auto, Reader};
    let mut wb = open_workbook_auto(path).map_err(|e| format!("xlsx: open: {e}"))?;
    let names = wb.sheet_names().to_owned();
    // Build the per-sheet body first, tracking how many sheets actually HAVE content -
    // sheet names can be empty or meaningless ("Sheet1"), so every sheet is labelled with a
    // 1-based INDEX too, and the summary reports content-bearing sheets (not just the total).
    let mut body = String::new();
    let mut with_content = 0usize;
    for (i, name) in names.iter().enumerate() {
        let idx = i + 1;
        let label = if name.trim().is_empty() {
            "(unnamed)".to_string()
        } else {
            format!("\"{name}\"")
        };
        let range = match wb.worksheet_range(name) {
            Ok(r) => r,
            Err(e) => {
                body.push_str(&format!("# Sheet {idx} {label}: unreadable ({e})\n\n"));
                continue;
            }
        };
        let (rows, cols) = (range.height(), range.width());
        if rows == 0 {
            body.push_str(&format!("# Sheet {idx} {label}: empty\n\n"));
            continue;
        }
        with_content += 1;
        // Header carries index + name + true size, so a truncated sheet still advertises
        // itself and is reachable by index even when the name is useless.
        body.push_str(&format!(
            "# Sheet {idx} {label}: {rows} rows × {cols} cols\n"
        ));
        for row in range.rows() {
            let cells: Vec<String> = row.iter().map(|c| c.to_string()).collect();
            body.push_str(&cells.join("\t"));
            body.push('\n');
        }
        body.push('\n');
    }
    let summary = format!(
        "# Workbook: {} sheet(s), {} with content - read one with read_file sheet=\"<name or index>\"\n\n",
        names.len(),
        with_content
    );
    Ok(format!("{summary}{body}").trim_end().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_by_extension() {
        assert!(matches!(
            DocKind::from_path(Path::new("a.pdf")),
            Some(DocKind::Pdf)
        ));
        assert!(matches!(
            DocKind::from_path(Path::new("a.DOCX")),
            Some(DocKind::Docx)
        ));
        assert!(matches!(
            DocKind::from_path(Path::new("a.xlsx")),
            Some(DocKind::Xlsx)
        ));
        assert!(matches!(
            DocKind::from_path(Path::new("a.xls")),
            Some(DocKind::Xlsx)
        ));
        assert!(DocKind::from_path(Path::new("a.txt")).is_none());
        assert!(DocKind::from_path(Path::new("noext")).is_none());
    }

    #[test]
    fn docx_xml_yields_paragraph_text() {
        let xml = "<w:p><w:r><w:t>Hello</w:t></w:r><w:r><w:t xml:space=\"preserve\"> world</w:t>\
</w:r></w:p><w:p><w:r><w:t>Line &amp; two &lt;ok&gt;</w:t></w:r></w:p>";
        assert_eq!(strip_docx_xml(xml), "Hello world\nLine & two <ok>");
    }

    #[test]
    fn docx_tab_and_break_map_through() {
        let xml = "<w:p><w:r><w:t>a</w:t><w:tab/><w:t>b</w:t><w:br/><w:t>c</w:t></w:r></w:p>";
        assert_eq!(strip_docx_xml(xml), "a\tb\nc");
    }

    #[test]
    fn unescape_does_not_double_decode() {
        // A literal "&lt;" in the source (written as &amp;lt;) must stay "&lt;", not "<".
        assert_eq!(unescape_xml("&amp;lt;"), "&lt;");
    }
}
