Multi-File MIMO Workflow
========================

This guide explains the new multi-file Multi-Input Multi-Output (MIMO) workflow for analyzing balance data across multiple stimulus-response pairs.

Problem Statement
-----------------

Previously, analyzing multiple stimuli and responses required awkward workarounds:

- **Inefficient data loading**: Same CSV file loaded multiple times
- **Error-prone manual pairing**: Creating sr_data objects manually
- **No cross-file pairing**: Could not combine stimulus from file A with response from file B

The new API solves these issues by supporting flexible, explicit stimulus-response pairing across multiple files.

Quick Start
-----------

**Single-file workflow (original API):**

.. code-block:: python

    sr_dict = bp.get_sr_data(
        filename='data.csv',
        body_height_m=1.75,
        sr_config=config,
        preproc_config=preproc,
    )
    # Returns: {'stim_pitch': sr_data, ...}

**Multi-file MIMO workflow (new):**

.. code-block:: python

    configs = [
        ('session1.csv', bp.AnaropiaSRDataConfig(
            stimulus_name='stim_pitch',
            response_name=bp.COM_LEGACY_AP,
        )),
        ('session2.csv', bp.AnaropiaSRDataConfig(
            stimulus_name='stim_roll',
            response_name=bp.COM_LEGACY_ML,
        )),
    ]
    
    sr_dict = bp.get_sr_data_multi(
        body_height_m=1.75,
        preproc_config=preproc,
        configs=configs,
    )
    # Returns: {('session1.csv', 'stim_pitch'): sr_data, 
    #           ('session2.csv', 'stim_roll'): sr_data}
    
    # Visualize all pairs
    fig, _ = bp.plot_datacheck_multi(sr_dict, output_dir='results/')
    
    # Fit jointly
    model1 = bp.Peterka18(..., data_exp=sr_dict[('session1.csv', 'stim_pitch')])
    model2 = bp.Peterka18(..., data_exp=sr_dict[('session2.csv', 'stim_roll')])
    multi = bp.MultiModel([model1, model2])
    multi.fit()

Core API
--------

Two complementary functions handle single-file and multi-file workflows:

**get_sr_data()** — Single-file loading (original API)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    sr_dict = bp.get_sr_data(
        filename='data.csv',
        body_height_m=1.75,
        sr_config=config,
        preproc_config=preproc,
    )

Parameters: ``filename``, ``body_height_m``, ``sr_config``, ``preproc_config``

Returns: ``{stimulus_col: sr_data, ...}``

**get_sr_data_multi()** — Multi-file MIMO loading (new)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    sr_dict = bp.get_sr_data_multi(
        body_height_m=1.75,
        preproc_config=preproc,
        configs=[
            (filename1, sr_config1),
            (filename2, sr_config2),
            ...
        ]
    )

Parameters: ``body_height_m``, ``preproc_config``, ``configs``

Returns: ``{(filename, stimulus_col): sr_data, ...}``

Shared Logic
^^^^^^^^^^^^

Both functions internally use ``_load_and_extract_sr_data()`` helper to:

- Load each file once
- Extract and preprocess response (shared within file)
- Extract and preprocess each stimulus column
- Create sr_data objects

This ensures:

- DRY code (no duplication)
- Consistent behavior across APIs
- Efficient data loading

plot_datacheck_multi()
~~~~~~~~~~~~~~~~~~~~~~

New function for visualizing multiple stimulus-response pairs.

.. code-block:: python

    fig, _ = bp.plot_datacheck_multi(
        sr_dict,
        output_dir='results/datacheck_plots/',
        save=True,
    )

Features:

- One row per stimulus-response pair
- Four columns: stimulus(time), response(time), stimulus(freq), response(freq)
- Shows mean (solid line) + individual cycles (faint) for 2D data
- Frequency domain shows FFT magnitude
- Labels show filename:stimulus_col for clarity
- Saves as PNG (``datacheck_multi.png``)

Use Cases
---------

Single-Stimulus, Single-Response
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use single-file API (unchanged):

.. code-block:: python

    sr_dict = bp.get_sr_data('data.csv', 1.75, sr_config, preproc)
    # {'stim_pitch': sr_data}

Multi-Stimulus, Single-Response (Same File)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use single-file API with tuple of stimulus columns:

.. code-block:: python

    sr_config = bp.AnaropiaSRDataConfig(
        stimulus_name=('stim_pitch', 'stim_roll'),  # Multiple stimuli
        response_name=bp.COM_LEGACY_AP,  # Single response
    )
    sr_dict = bp.get_sr_data('data.csv', 1.75, sr_config, preproc)
    # {'stim_pitch': sr_data, 'stim_roll': sr_data}  # Shared response

Multi-Stimulus, Multi-Response (Different Files)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use multi-file API with separate configs per (stimulus, response) pair:

.. code-block:: python

    configs = [
        ('session1.csv', bp.AnaropiaSRDataConfig(
            stimulus_name='stim_pitch',
            response_name=bp.COM_LEGACY_AP,  # AP response for pitch
        )),
        ('session2.csv', bp.AnaropiaSRDataConfig(
            stimulus_name='stim_pitch',
            response_name=bp.COM_LEGACY_ML,  # ML response for pitch (different file)
        )),
        ('session3.csv', bp.AnaropiaSRDataConfig(
            stimulus_name='stim_roll',
            response_name=bp.COM_LEGACY_ML,  # ML response for roll
        )),
    ]
    sr_dict = bp.get_sr_data_multi(body_height_m=1.75, preproc_config=preproc, configs=configs)
    # {('session1.csv', 'stim_pitch'): sr_data_ap,
    #  ('session2.csv', 'stim_pitch'): sr_data_ml,
    #  ('session3.csv', 'stim_roll'): sr_data}

Same Stimulus, Multiple Responses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use multi-file API to extract the same stimulus with different response types:

.. code-block:: python

    configs = [
        ('data.csv', bp.AnaropiaSRDataConfig(
            stimulus_name='stim_pitch',
            response_name=bp.COM_LEGACY_AP,  # AP response
        )),
        ('data.csv', bp.AnaropiaSRDataConfig(
            stimulus_name='stim_pitch',
            response_name=bp.COM_LEGACY_ML,  # ML response (same file, different config)
        )),
    ]
    sr_dict = bp.get_sr_data_multi(body_height_m=1.75, preproc_config=preproc, configs=configs)
    # {('data.csv', 'stim_pitch'): sr_data_ap,
    #  ('data.csv', 'stim_pitch'): sr_data_ml}  # Different responses, same stimulus

Advanced: Flexible Cross-File Pairing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can combine stimulus from one file with response from another:

.. code-block:: python

    advanced_configs = [
        # Load pitch stimulus, extract AP response from SAME file
        (
            'session_A.csv',
            bp.AnaropiaSRDataConfig(
                stimulus_name='stim_pitch',
                response_name=bp.COM_LEGACY_AP,
            ),
        ),
        # Load DIFFERENT stimulus, extract response from DIFFERENT file
        (
            'session_B.csv',
            bp.AnaropiaSRDataConfig(
                stimulus_name='stim_roll',
                response_name=bp.COM_LEGACY_ML,  # Different response source!
            ),
        ),
    ]
    
    sr_dict = bp.get_sr_data_multi(body_height_m=1.75, preproc_config=preproc, configs=advanced_configs)

**Use case:** Cross-session analysis

- Session A: Good pitch stimulus, poor ML response
- Session B: Poor pitch stimulus, good ML response
- → Load pitch from A, ML response from B, fit jointly!

**Constraint:** Stimulus and response columns must come from SAME FILE within a single sr_config.

Multi-Model Fitting Workflow
-----------------------------

After loading sr_data with multi-file API:

.. code-block:: python

    # 1. Load data
    sr_dict = bp.get_sr_data_multi(body_height_m=1.75, preproc_config=preproc, configs=configs)
    
    # 2. Visualize
    fig, _ = bp.plot_datacheck_multi(sr_dict)
    fig.show()
    
    # 3. Create models (one per stimulus-response pair)
    models = []
    for key, sr_obj in sr_dict.items():
        model = bp.Peterka18(mass_kg=80, height_m=1.75, data_exp=sr_obj)
        models.append(model)
    
    # 4. Create MultiModel
    multi = bp.MultiModel(models)
    
    # 5. Fit (optimizes shared parameters across all stimuli)
    theta_fit, fit_output = multi.fit()
    
    # 6. Inspect results
    print("Fitted parameters:", multi.params.values(only_free=False))
    print("Fit error:", fit_output.fun)

Best Practices
--------------

1. **Configuration Organization**
   
   - Use ONE ``AnaropiaSRDataConfig`` per (file, stimulus-response pair)
   - ``preproc_config`` is SHARED (same for all files)
   - Different files → different ``sr_config`` for clarity
   
2. **Data Organization**
   
   - Store configs in a list to avoid repetition
   - Name files consistently (e.g., ``session1.csv``, ``session2.csv``)
   - Document which columns contain stimulus/response in each file

3. **Visualization**
   
   - Always call ``plot_datacheck_multi()`` before fitting
   - Check for:
     
     - Correct stimulus and response signals
     - Data quality (noise, artifacts)
     - Frequency content (SNR, coherence across pairs)
     - Cycle consistency across stimulus conditions

4. **Fitting**
   
   - Use ``MultiModel`` to fit across multiple stimuli
   - Body parameters (height, mass) should be SAME for all models
   - Stimulus-specific parameters (gains) should have different names
   - Check that error decreases across all models

5. **Validation**
   
   - Plot fitted FRF vs experimental FRF for each stimulus
   - Check parameter distributions (physically plausible?)
   - Compare to single-stimulus fits (should be similar)
   - Report error/coherence per stimulus

Troubleshooting
---------------

❌ **Problem:** "Parameter 'X' not found in MultiModelParameterSet"

   **Solution:** Check that ``multimodel_name`` is set consistently in all models.
   Different models must use the same parameter names for shared parameters.

❌ **Problem:** Fit fails or converges poorly

   **Solution:** Check ``plot_datacheck_multi()`` output:
   
   - Are all signals present and reasonable?
   - Are frequency ranges appropriate?
   - Try different initial parameter values
   - Consider fitting individual stimuli first, then multi-fit

❌ **Problem:** Stimulus from file A, response from file B fails

   **Solution:** Stimulus and response must be from SAME FILE within one ``sr_config``.
   Use separate ``sr_config`` objects for different file combinations.

❌ **Problem:** Data has different sampling rates in different files

   **Solution:** Preprocessing is shared (same for all files).
   Resample all files to the same rate BEFORE loading, or use separate ``get_sr_data()`` calls.

See Also
--------

- :func:`balancepy.anaropia_project.get_sr_data` — Single-file loading
- :func:`balancepy.anaropia_project.get_sr_data_multi` — Multi-file MIMO loading
- :func:`balancepy.anaropia.plot_datacheck_multi` — Multi-stimulus visualization
- :class:`balancepy.model_sim.multi_model.MultiModel` — Joint fitting across stimuli
- :doc:`models` — Model documentation
