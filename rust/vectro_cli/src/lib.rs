use std::io::{BufRead, BufReader, Write};
use indicatif::{ProgressBar, ProgressStyle};

pub fn compress_stream(input: &str, output: &str, quantize: bool) -> anyhow::Result<usize> {
    use crossbeam_channel::{bounded, Sender, Receiver};
    use std::thread;

    let header = b"VECTRO+STREAM1\n";
    let infile = std::fs::File::open(input)?;
    let reader = BufReader::new(infile);

    let outfile = std::fs::File::create(output)?;
    let writer_buf = std::io::BufWriter::new(outfile);

    // channels
    let (item_tx, item_rx): (Sender<vectro_lib::Embedding>, Receiver<vectro_lib::Embedding>) = bounded(1024);
    let (bytes_tx, bytes_rx): (Sender<Vec<u8>>, Receiver<Vec<u8>>) = bounded(1024);

    // writer thread (non-quantized path will spawn writer now; quantized path spawns writer after tables computed)
    let out_clone = output.to_string();
    let qheader = b"VECTRO+QSTREAM1\n";
    let mut writer_handle_opt = None;
    // prepare worker handles container
    let mut worker_handles: Vec<std::thread::JoinHandle<()>> = Vec::new();
    if !quantize {
        let mut w = writer_buf;
        let rx_for_writer = bytes_rx.clone();
        let out_for_writer = out_clone.clone();
        let header_local = *header;
        let handle = thread::spawn(move || -> anyhow::Result<()> {
            w.write_all(&header_local)?;
            let mut written = 0usize;
            while let Ok(bytes) = rx_for_writer.recv() {
                let len = (bytes.len() as u32).to_le_bytes();
                w.write_all(&len)?;
                w.write_all(&bytes)?;
                written += 1;
            }
            w.flush()?;
            eprintln!("wrote {} entries to {}", written, out_for_writer);
            Ok(())
        });
        writer_handle_opt = Some(handle);
        // spawn workers for non-quantized path
        let workers = num_cpus::get().max(1);
        for _ in 0..workers {
            let r = item_rx.clone();
            let tx = bytes_tx.clone();
            worker_handles.push(thread::spawn(move || {
                while let Ok(e) = r.recv() {
                    if let Ok(bytes) = bincode::serialize(&e) {
                        let _ = tx.send(bytes);
                    }
                }
            }));
        }
    }

    // don't spawn workers yet; will spawn depending on quantize mode

    // progress bar
    let pb = ProgressBar::new_spinner();
    pb.set_style(ProgressStyle::with_template("{spinner} {msg}").unwrap());
    pb.enable_steady_tick(std::time::Duration::from_millis(80));
    if quantize {
        pb.set_message("parsing and computing quant tables...");
    } else {
        pb.set_message("compressing (streaming bincode)...");
    }

    // reader: parse lines and collect embeddings
    let mut parsed = 0usize;
    // collect embeddings when quantizing
    let mut collected_embeddings: Vec<vectro_lib::Embedding> = Vec::new();
    for line in reader.lines().map_while(Result::ok) {
        let line = line.trim();
        if line.is_empty() { continue; }

        // try JSON
        let mut pushed = false;
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
            if let (Some(id), Some(vec)) = (val.get("id"), val.get("vector")) {
                if let (Some(id_str), Some(arr)) = (id.as_str(), vec.as_array()) {
                    let mut v = Vec::with_capacity(arr.len());
                    for x in arr { if let Some(flt) = x.as_f64() { v.push(flt as f32); } }
                    let emb = vectro_lib::Embedding::new(id_str, v.clone());
                    if quantize { collected_embeddings.push(emb.clone()); } else { let _ = item_tx.send(emb); }
                    parsed += 1;
                    pushed = true;
                }
            }
        }
        if !pushed {
            // CSV
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 2 {
                let id = parts[0].to_string();
                let mut v = Vec::new();
                for p in &parts[1..] { if let Ok(f) = p.trim().parse::<f32>() { v.push(f); } }
                let emb = vectro_lib::Embedding::new(id, v.clone());
                if quantize { collected_embeddings.push(emb.clone()); } else { let _ = item_tx.send(emb); }
                parsed += 1;
            }
        }

        if parsed % 100 == 0 { pb.set_message(format!("parsed {} entries", parsed)); }
    }

    if quantize {
        // compute tables using vectro_lib::search::quant::quantize_dataset
        let vectors: Vec<Vec<f32>> = collected_embeddings.iter().map(|e| e.vector.clone()).collect();
        let (tables, _qvecs) = vectro_lib::search::quant::quantize_dataset(&vectors);
        // serialize tables to bincode
        let tables_blob = bincode::serialize(&tables)?;

        // write header + tables to file, then spawn writer thread to append entries
        {
            // overwrite/create file and write header+tables
            let mut f = std::fs::File::create(output)?;
            let mut w = std::io::BufWriter::new(&mut f);
            w.write_all(qheader)?;
            let table_count = (tables.len() as u32).to_le_bytes();
            let dim = (if !tables.is_empty() { tables.len() as u32 } else { 0u32 }).to_le_bytes();
            let tables_len = (tables_blob.len() as u32).to_le_bytes();
            w.write_all(&table_count)?;
            w.write_all(&dim)?;
            w.write_all(&tables_len)?;
            w.write_all(&tables_blob)?;
            w.flush()?;
        }

        // spawn writer that appends entries
        let outfile = std::fs::OpenOptions::new().append(true).open(output)?;
        let writer_buf = std::io::BufWriter::new(outfile);
        let out_clone2 = out_clone.clone();
        let handle = thread::spawn(move || -> anyhow::Result<()> {
            let mut w = writer_buf;
            let mut written = 0usize;
            while let Ok(bytes) = bytes_rx.recv() {
                let len = (bytes.len() as u32).to_le_bytes();
                w.write_all(&len)?;
                w.write_all(&bytes)?;
                written += 1;
            }
            w.flush()?;
            eprintln!("wrote {} entries to {}", written, out_clone2);
            Ok(())
        });
        writer_handle_opt = Some(handle);

        // spawn workers to quantize embeddings
        let workers = num_cpus::get().max(1);
        use crossbeam_channel::bounded;
        let (item_tx2, item_rx2) = bounded::<vectro_lib::Embedding>(1024);
        // worker threads
        for _ in 0..workers {
            let r = item_rx2.clone();
            let tx = bytes_tx.clone();
            let tables = tables.clone();
            worker_handles.push(thread::spawn(move || {
                while let Ok(e) = r.recv() {
                    // quantize vector
                    let qv: Vec<u8> = e.vector.iter().enumerate().map(|(i, &x)| tables[i].quantize(x)).collect();
                    let rec = (e.id.clone(), qv);
                    if let Ok(bytes) = bincode::serialize(&rec) {
                        let _ = tx.send(bytes);
                    }
                }
            }));
        }

        // feed collected embeddings into item_tx2
        for emb in collected_embeddings {
            let _ = item_tx2.send(emb);
        }
        drop(item_tx2);

        // wait for workers
        drop(bytes_tx);
        for h in worker_handles { let _ = h.join(); }
        // wait for writer
        if let Some(h) = writer_handle_opt { let _ = h.join(); }

    } else {
        // close item_tx to signal workers to finish
        drop(item_tx);
        // wait for workers
        drop(bytes_tx);
        for h in worker_handles { let _ = h.join(); }
        // wait for writer
        if let Some(h) = writer_handle_opt { let _ = h.join(); }
    }
    if quantize {
    // If quantized, show a short summary including table count (attempt to read tables from file)
    // variable intentionally unused; underscore prefix to silence warnings
    let _table_count = 0usize;
        if let Ok(mut f) = std::fs::File::open(output) {
            use std::io::Read;
            let mut hdr = vec![0u8; 16];
            let _ = f.read(&mut hdr);
            // crude: read table_count at offset header.len()
            // header 'VECTRO+QSTREAM1\n' length is 14
            if hdr.len() >= 16 {
                // no-op; we will just display quantized
            }
        }
        pb.finish_with_message(format!("wrote {} quantized entries to {}", parsed, output));
    } else {
        pb.finish_with_message(format!("wrote {} entries to {}", parsed, output));
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn compress_small_file() {
        let tmp_in = NamedTempFile::new().unwrap();
        let in_path = tmp_in.path().to_str().unwrap().to_string();
        std::fs::write(&in_path, r#"{"id":"one","vector":[1.0,0.0]}
{"id":"two","vector":[0.0,1.0]}"#).unwrap();

        let tmp_out = NamedTempFile::new().unwrap();
        let out_path = tmp_out.path().to_str().unwrap().to_string();

        let n = compress_stream(&in_path, &out_path, false).expect("compress");
        assert_eq!(n, 2);

        let ds = vectro_lib::EmbeddingDataset::load(&out_path).expect("load");
        assert_eq!(ds.len(), 2);
    }

    #[test]
    fn compress_quantized() {
        let tmp_in = NamedTempFile::new().unwrap();
        let in_path = tmp_in.path().to_str().unwrap().to_string();
        std::fs::write(&in_path, r#"{"id":"one","vector":[1.0,2.0,3.0]}
{"id":"two","vector":[4.0,5.0,6.0]}
{"id":"three","vector":[7.0,8.0,9.0]}"#).unwrap();

        let tmp_out = NamedTempFile::new().unwrap();
        let out_path = tmp_out.path().to_str().unwrap().to_string();

        let n = compress_stream(&in_path, &out_path, true).expect("compress quantized");
        assert_eq!(n, 3);

        let ds = vectro_lib::EmbeddingDataset::load(&out_path).expect("load");
        assert_eq!(ds.len(), 3);
        // Quantized embeddings may not preserve order
        let ids: Vec<&str> = ds.embeddings.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&"one"));
        assert!(ids.contains(&"two"));
        assert!(ids.contains(&"three"));
    }

    #[test]
    fn compress_csv_format() {
        let tmp_in = NamedTempFile::new().unwrap();
        let in_path = tmp_in.path().to_str().unwrap().to_string();
        std::fs::write(&in_path, "id1,1.0,2.0,3.0\nid2,4.0,5.0,6.0\n").unwrap();

        let tmp_out = NamedTempFile::new().unwrap();
        let out_path = tmp_out.path().to_str().unwrap().to_string();

        let n = compress_stream(&in_path, &out_path, false).expect("compress csv");
        assert_eq!(n, 2);

        let ds = vectro_lib::EmbeddingDataset::load(&out_path).expect("load");
        assert_eq!(ds.len(), 2);
    }

    #[test]
    fn compress_with_empty_lines() {
        let tmp_in = NamedTempFile::new().unwrap();
        let in_path = tmp_in.path().to_str().unwrap().to_string();
        std::fs::write(&in_path, r#"
{"id":"one","vector":[1.0,0.0]}

{"id":"two","vector":[0.0,1.0]}

"#).unwrap();

        let tmp_out = NamedTempFile::new().unwrap();
        let out_path = tmp_out.path().to_str().unwrap().to_string();

        let n = compress_stream(&in_path, &out_path, false).expect("compress");
        assert_eq!(n, 2);
    }
}
