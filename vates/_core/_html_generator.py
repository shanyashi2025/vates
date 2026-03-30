from datetime import datetime
import os
from typing import Dict, Any

def generate_runlog_html(runlog: Dict[str, Any]) -> str:
    """
    Convert a runlog dictionary to a beautiful HTML visualization.
    
    Args:
        runlog: Dictionary containing model run information
    
    Returns:
        HTML content as string
    """
    
    def calculate_duration(start_str: str, end_str: str) -> str:
        """Calculate duration between start and end times"""
        start = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
        end = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
        sec = (end - start).total_seconds()

        if sec < 60:
            return f"~{int(sec)} second{'s' if sec != 1 else ''}"
        elif sec < 3600:
            minutes = round(sec / 60, 1)
            return f"~{minutes} minute{'s' if int(minutes) != 1 else ''}"
        else:
            hours = round(sec / 3600, 2)
            return f"~{hours} hour{'s' if int(hours) != 1 else ''}"
    
    def get_file_info(file_path: str) -> tuple:
        """Get file modification time and size"""
        try:
            if os.path.exists(file_path):
                stat_info = os.stat(file_path)
                file_mtime = datetime.fromtimestamp(stat_info.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                size_bytes = stat_info.st_size
                
                # Format file size
                if size_bytes < 1024 * 1024:
                    size_str = f"{size_bytes / 1024:.1f} KB"
                elif size_bytes < 1024 * 1024 * 1024:
                    size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
                else:
                    size_str = f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
                
                return file_mtime, size_str
            else:
                return "File not found", "N/A"
        except Exception as e:
            return "Unknown", "Unknown"
    
    # Extract data from runlog
    model_name = runlog['model_name']
    model_desc = runlog['model_desc']
    exe_start_time = runlog['exe_start_time']
    exe_end_time = runlog['exe_end_time']
    failed_simulations = runlog.get('failed_simulations',[])
    settings = runlog['setting']
    input_files = runlog['input_files']
    proj_result_files = runlog['proj_result_files']
    stoch_result_files = runlog['stoch_result_files']
    other_result_files = runlog['other_result_files']
    
    # Calculate execution duration
    exe_duration = calculate_duration(exe_start_time, exe_end_time)
    
    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Run Log - {model_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        
        .header .subtitle {{
            opacity: 0.9;
            font-size: 1.1em;
            margin-top: 10px;
        }}
        
        .content {{
            padding: 30px;
        }}
        
        .section {{
            margin-bottom: 30px;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .section-header {{
            background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
            color: white;
            padding: 15px 20px;
            font-size: 1.3em;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.3s ease;
        }}
        
        .section-header:hover {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        }}
        
        .section-content {{
            background: #f8f9fa;
            padding: 20px;
        }}
        
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }}
        
        .info-item {{
            background: white;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #3498db;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        
        .info-label {{
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 5px;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .info-value {{
            color: #34495e;
            font-family: 'Consolas', 'Monaco', monospace;
            background: #ecf0f1;
            padding: 8px;
            border-radius: 4px;
            word-break: break-all;
        }}
        
        .file-list {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        
        .file-item {{
            background: white;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #e74c3c;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            display: grid;
            grid-template-columns: 40px 1fr auto auto;
            gap: 15px;
            align-items: center;
        }}
        
        .file-item.result {{
            border-left-color: #27ae60;
        }}
        
        .file-number {{
            font-weight: bold;
            color: #3498db;
            font-size: 1.1em;
        }}
        
        .file-details {{
            display: flex;
            flex-direction: column;
        }}
        
        .file-name {{
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 2px;
        }}
        
        .file-path {{
            font-family: 'Consolas', 'Monaco', monospace;
            color: #7f8c8d;
            font-size: 0.85em;
        }}
        
        .file-mod-time {{
            font-size: 0.8em;
            color: #95a5a6;
            text-align: center;
        }}
        
        .file-size {{
            font-size: 0.8em;
            color: #95a5a6;
            font-weight: 500;
            text-align: right;
        }}
        
        .collapsible {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }}
        
        .collapsible.active {{
            max-height: 2000px;
        }}
        
        .toggle-btn {{
            float: right;
            font-size: 1.2em;
            transition: transform 0.3s ease;
        }}
        
        .toggle-btn.rotated {{
            transform: rotate(180deg);
        }}
        
        .duration {{
            background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
            color: white;
            padding: 10px 20px;
            border-radius: 20px;
            display: inline-block;
            font-weight: 600;
            margin-top: 10px;
        }}
        
        .list-value {{
            color: #34495e;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                margin: 10px;
                border-radius: 8px;
            }}
            
            .content {{
                padding: 20px;
            }}
            
            .info-grid {{
                grid-template-columns: 1fr;
            }}
            
            .file-list {{
                flex-direction: column;
            }}
            
            .file-item {{
                grid-template-columns: 30px 1fr;
                gap: 10px;
            }}
            
            .file-mod-time,
            .file-size {{
                display: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Model Run Log</h1>
            <div class="subtitle">{model_name.replace('_', ' ').title()} Execution Flash Report</div>
        </div>
        
        <div class="content">
            <!-- Basic Info Section -->
            <div class="section">
                <div class="section-header" onclick="toggleSection('basic-info')">
                    📊 Basic Information
                    <span class="toggle-btn" id="basic-info-btn">▼</span>
                </div>
                <div class="section-content collapsible active" id="basic-info-content">
                    <div class="info-grid">
                        <div class="info-item" style="grid-column: 1 / 1;">
                            <div class="info-label">Model Name</div>
                            <div class="info-value">{model_name}</div>
                        </div>
                        <div class="info-item" style="grid-column: 2 / -1;">
                            <div class="info-label">Model Description</div>
                            <div class="info-value">{model_desc}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Execution Start Time</div>
                            <div class="info-value">{exe_start_time}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Execution End Time</div>
                            <div class="info-value">{exe_end_time}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Execution Duration</div>
                            <div class="duration">{exe_duration}</div>
                        </div>
                    </div>
                </div>
            </div>"""
    
    # Settings Section
    if settings:
        html_content += """
            <!-- Settings Section -->
            <div class="section">
                <div class="section-header" onclick="toggleSection('settings')">
                    ⚙️ Run Setting
                    <span class="toggle-btn" id="settings-btn">▼</span>
                </div>
                <div class="section-content collapsible active" id="settings-content">
                    <div class="info-grid">"""
        
        for key, value in settings.items():
            if isinstance(value, list):
                value_str = ', '.join(str(v) for v in value)
            else:
                value_str = str(value)
            
            # Special formatting for specific keys
            label = key.replace('_', ' ').title()
            html_content += f"""
                        <div class="info-item">
                            <div class="info-label">{label}</div>
                            <div class="info-value">{value_str}</div>
                        </div>"""
        
        html_content += """
                    </div>
                </div>
            </div>"""
    
    # Input Section
    if input_files:
        html_content += """
            <!-- Input Section -->
            <div class="section">
                <div class="section-header" onclick="toggleSection('input')">
                    📁 Input Files
                    <span class="toggle-btn" id="input-btn">▼</span>
                </div>
                <div class="section-content collapsible active" id="input-content">
                    <div class="file-list">"""

        for i, (key, path) in enumerate(input_files.items(), 1):
            mod_time, file_size = get_file_info(path)
            html_content += f"""
                        <div class="file-item">
                            <div class="file-number">{i}</div>
                            <div class="file-details">
                                <div class="file-name">{key}</div>
                                <div class="file-path">{path}</div>
                            </div>
                            <div class="file-mod-time">{mod_time}</div>
                            <div class="file-size">{file_size}</div>
                        </div>"""

        html_content += """
                    </div>
                </div>
            </div>"""

    # Projection Results Section
    if proj_result_files:
        html_content += """
            <!-- Projection Results Section -->
            <div class="section">
                <div class="section-header" onclick="toggleSection('projres')">
                    📈 Output - Projection Results
                    <span class="toggle-btn" id="projres-btn">▼</span>
                </div>
                <div class="section-content collapsible active" id="projres-content">
                    <div class="file-list">"""

        for i, (key, path) in enumerate(proj_result_files.items(), 1):
            mod_time, file_size = get_file_info(path)
            html_content += f"""
                        <div class="file-item result">
                            <div class="file-number">{i}</div>
                            <div class="file-details">
                                <div class="file-name">{key}</div>
                                <div class="file-path">{path}</div>
                            </div>
                            <div class="file-mod-time">{mod_time}</div>
                            <div class="file-size">{file_size}</div>
                        </div>"""

        html_content += """
                    </div>
                </div>
            </div>"""

    # Stochastic Results Section
    if stoch_result_files:
        html_content += """
            <!-- Stochastic Results Section -->
            <div class="section">
                <div class="section-header" onclick="toggleSection('stochres')">
                    📈 Output - Stochastic Results
                    <span class="toggle-btn" id="stochres-btn">▼</span>
                </div>
                <div class="section-content collapsible active" id="stochres-content">
                    <div class="file-list">"""

        for i, (key, path) in enumerate(stoch_result_files.items(), 1):
            mod_time, file_size = get_file_info(path)
            html_content += f"""
                        <div class="file-item result">
                            <div class="file-number">{i}</div>
                            <div class="file-details">
                                <div class="file-name">{key}</div>
                                <div class="file-path">{path}</div>
                            </div>
                            <div class="file-mod-time">{mod_time}</div>
                            <div class="file-size">{file_size}</div>
                        </div>"""

        html_content += """
                    </div>
                </div>
            </div>"""

    # Failed Simulations Section
    if failed_simulations:
        html_content += """
            <!-- Failed Simulations Section -->
            <div class="section">
                <div class="section-header" onclick="toggleSection('failsim')">
                    ⚠️ Failed Simulation
                    <span class="toggle-btn" id="failsim-btn">▼</span>
                </div>
                <div class="section-content collapsible active" id="failsim-content">
                    <div class="file-list">"""

        for sim, err in failed_simulations:
            html_content += f"""
                        <div class="simulation result">
                            <div class="file-number">{sim}</div>
                            <div class="file-path">{err}</div>
                        </div>"""

        html_content += """
                    </div>
                </div>
            </div>"""

    # Other Results Section
    if other_result_files:
        html_content += """
            <!-- Other Results Section -->
            <div class="section">
                <div class="section-header" onclick="toggleSection('otherres')">
                    📈 Output - Other
                    <span class="toggle-btn" id="otherres-btn">▼</span>
                </div>
                <div class="section-content collapsible active" id="otherres-content">
                    <div class="file-list">"""

        for i, (key, path) in enumerate(other_result_files.items(), 1):
            mod_time, file_size = get_file_info(path)
            html_content += f"""
                        <div class="file-item result">
                            <div class="file-number">{i}</div>
                            <div class="file-details">
                                <div class="file-name">{key}</div>
                                <div class="file-path">{path}</div>
                            </div>
                            <div class="file-mod-time">{mod_time}</div>
                            <div class="file-size">{file_size}</div>
                        </div>"""

        html_content += """
                    </div>
                </div>
            </div>"""

    # Close HTML
    html_content += """
        </div>
    </div>
    
    <script>
        function toggleSection(sectionId) {
            const content = document.getElementById(sectionId + '-content');
            const btn = document.getElementById(sectionId + '-btn');
            
            content.classList.toggle('active');
            btn.classList.toggle('rotated');
        }
    </script>
</body>
</html>"""

    return html_content