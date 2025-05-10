ValueError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/mount/src/fundflow_advisor/app.py", line 157, in <module>
    main()
File "/mount/src/fundflow_advisor/app.py", line 115, in main
    df = extract_tables_from_pdf(buf)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/mount/src/fundflow_advisor/app.py", line 32, in extract_tables_from_pdf
    aligned = [df.reindex(columns=all_cols) for df in raw_tables]
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.12/site-packages/pandas/core/frame.py", line 5378, in reindex
    return super().reindex(
           ^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.12/site-packages/pandas/core/generic.py", line 5610, in reindex
    return self._reindex_axes(
           ^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.12/site-packages/pandas/core/generic.py", line 5633, in _reindex_axes
    new_index, indexer = ax.reindex(
                         ^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.12/site-packages/pandas/core/indexes/base.py", line 4429, in reindex
    raise ValueError("cannot reindex on an axis with duplicate labels")
