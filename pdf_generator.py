"""
Modulo para generacion de informes PDF del diagnostico de dengue grave
"""

from datetime import datetime
from fpdf import FPDF


def generar_pdf_ricky(paciente_info, sintomas_seleccionados, resultados, datos_historicos):
    """
    Genera un PDF con contenido clinico rico
    """
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    
    # ============================================================
    # TITULO
    # ============================================================
    pdf.cell(0, 10, 'INFORME CLINICO - EVALUACION DENGUE', 0, 1, 'C')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 5, f'Fecha: {datetime.now().strftime("%d/%m/%Y %H:%M")}', 0, 1, 'C')
    pdf.cell(0, 5, f'ID Informe: {datetime.now().strftime("%Y%m%d%H%M%S")}', 0, 1, 'C')
    pdf.ln(8)
    
    # ============================================================
    # RIESGO (con mensaje clinico)
    # ============================================================
    nivel = resultados['riesgo']
    pdf.set_font('Arial', 'B', 14)
    
    if nivel == "ALTO":
        pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 10, 'RIESGO ALTO', 0, 1, 'C')
        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 6, 'El paciente presenta una combinacion de sintomas y factores de riesgo que sugieren alta probabilidad de evolucion a dengue grave.')
    elif nivel == "MODERADO":
        pdf.set_text_color(255, 165, 0)
        pdf.cell(0, 10, 'RIESGO MODERADO', 0, 1, 'C')
        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 6, 'El paciente presenta factores de riesgo que requieren atencion y seguimiento clinico.')
    else:
        pdf.set_text_color(0, 128, 0)
        pdf.cell(0, 10, 'RIESGO BAJO', 0, 1, 'C')
        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 6, 'El paciente no presenta factores de riesgo significativos en esta evaluacion.')
    
    pdf.ln(5)
    
    # ============================================================
    # 1. DATOS DEL PACIENTE
    # ============================================================
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, '1. DATOS DEL PACIENTE', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 6, f'Edad: {paciente_info["edad"]} anos', 0, 1)
    pdf.cell(0, 6, f'Sexo: {paciente_info["sexo"]}', 0, 1)
    pdf.cell(0, 6, f'Estrato socioeconomico: {paciente_info["estrato"]}', 0, 1)
    
    # Contexto del estrato
    if paciente_info['estrato'] <= 2:
        pdf.multi_cell(0, 5, 'Contexto: Los estratos 1 y 2 concentran el 85% de los casos historicos de dengue en la region.')
    elif paciente_info['estrato'] <= 4:
        pdf.multi_cell(0, 5, 'Contexto: Los estratos 3 y 4 presentan una incidencia moderada de dengue en la region.')
    else:
        pdf.multi_cell(0, 5, 'Contexto: Los estratos 5 y 6 presentan la menor incidencia historica de dengue en la region.')
    pdf.ln(3)
    
    # ============================================================
    # 2. SINTOMAS REPORTADOS
    # ============================================================
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, '2. SINTOMAS REPORTADOS', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    
    if sintomas_seleccionados:
        for s in sintomas_seleccionados:
            s_clean = s.replace('✅', '').strip()
            pdf.cell(0, 6, f'- {s_clean}', 0, 1)
        
        # Detectar signos de alarma
        signos_alarma = ["Dolor abdominal", "Vomito", "Somnolencia", "Hepatomegalia"]
        alarma_presentes = [s for s in sintomas_seleccionados if s in signos_alarma]
        if alarma_presentes:
            pdf.set_text_color(255, 0, 0)
            pdf.multi_cell(0, 5, f'ADVERTENCIA: Signos de alarma detectados: {", ".join(alarma_presentes)}')
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 5, 'Estos sintomas requieren atencion medica prioritaria.')
    else:
        pdf.cell(0, 6, 'No se reportaron sintomas clinicos', 0, 1)
    pdf.ln(3)
    
    # ============================================================
    # 3. RESULTADO DE LA EVALUACION (SIN NUMEROS TECNICOS)
    # ============================================================
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, '3. RESULTADO DE LA EVALUACION', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    
    # Nivel de riesgo
    if nivel == "ALTO":
        pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 6, 'Nivel de riesgo: ALTO', 0, 1)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 5, 'Conclusion clinica: El paciente presenta alta probabilidad de evolucion a dengue grave.')
        pdf.set_font('Arial', 'B', 10)
        pdf.multi_cell(0, 5, 'Recomendacion principal: HOSPITALIZACION INMEDIATA y evaluacion por personal medico especializado.')
    elif nivel == "MODERADO":
        pdf.set_text_color(255, 165, 0)
        pdf.cell(0, 6, 'Nivel de riesgo: MODERADO', 0, 1)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 5, 'Conclusion clinica: El paciente requiere observacion clinica y seguimiento.')
        pdf.set_font('Arial', 'B', 10)
        pdf.multi_cell(0, 5, 'Recomendacion principal: OBSERVACION CLINICA y reevaluacion en las proximas 24-48 horas.')
    else:
        pdf.set_text_color(0, 128, 0)
        pdf.cell(0, 6, 'Nivel de riesgo: BAJO', 0, 1)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 5, 'Conclusion clinica: El paciente puede recibir manejo ambulatorio.')
        pdf.set_font('Arial', 'B', 10)
        pdf.multi_cell(0, 5, 'Recomendacion principal: MANEJO AMBULATORIO con seguimiento de sintomas en casa.')
    
    pdf.set_font('Arial', '', 10)
    pdf.ln(3)
    
    # ============================================================
    # 4. CONTEXTO EPIDEMIOLOGICO
    # ============================================================
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, '4. CONTEXTO EPIDEMIOLOGICO REGIONAL EN LOS ÚLTIMOS AÑOS(2018 - 2026)', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 6, f'Total de casos analizados: {datos_historicos["total_casos"]:,}', 0, 1)
    pdf.cell(0, 6, f'Casos de dengue grave registrados: {datos_historicos["total_graves"]:,} ({datos_historicos["porcentaje_graves"]:.2f}%)', 0, 1)
    pdf.ln(2)
    
    pdf.cell(0, 6, 'Distribucion de casos por estrato:', 0, 1)
    for e in range(1, 7):
        casos = datos_historicos['casos_por_estrato'][e]
        porcentaje = (casos / datos_historicos['total_casos']) * 100
        pdf.cell(30, 5, f'E{e}:', 0, 0)
        pdf.cell(0, 5, f'{casos:,} casos ({porcentaje:.1f}%)', 0, 1)
    pdf.ln(2)
    
    pdf.multi_cell(0, 5, 'Los estratos bajos (E1 y E2) concentran el 85% de los casos historicos, lo que explica la mayor frecuencia de dengue en estas poblaciones.')
    pdf.ln(3)
    
    # ============================================================
    # 5. RECOMENDACIONES CLINICAS
    # ============================================================
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, '5. RECOMENDACIONES CLINICAS', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    
    # Recomendaciones generales
    pdf.multi_cell(0, 5, 'Recomendaciones generales:')
    pdf.cell(0, 5, '- Mantener hidratacion adecuada (suero oral o liquidos abundantes)', 0, 1)
    pdf.cell(0, 5, '- Evitar automedicacion con antiinflamatorios (ibuprofeno, aspirina)', 0, 1)
    pdf.cell(0, 5, '- Acudir a control medico si aparecen signos de alarma', 0, 1)
    pdf.cell(0, 5, '- Reposo relativo durante la fase aguda', 0, 1)
    pdf.ln(2)
    
    # Recomendaciones especificas por riesgo
    pdf.set_font('Arial', 'B', 10)
    if nivel == "ALTO":
        pdf.multi_cell(0, 5, 'Recomendaciones especificas para riesgo ALTO:')
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 5, '- Hospitalizacion inmediata en centro de salud', 0, 1)
        pdf.cell(0, 5, '- Monitoreo de signos vitales cada 4 horas', 0, 1)
        pdf.cell(0, 5, '- Hemograma seriado cada 24-48 horas', 0, 1)
        pdf.cell(0, 5, '- Evaluacion por infectologia', 0, 1)
    elif nivel == "MODERADO":
        pdf.multi_cell(0, 5, 'Recomendaciones especificas para riesgo MODERADO:')
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 5, '- Observacion clinica por 24 horas', 0, 1)
        pdf.cell(0, 5, '- Control de signos vitales cada 8 horas', 0, 1)
        pdf.cell(0, 5, '- Re-evaluacion en 24-48 horas', 0, 1)
        pdf.cell(0, 5, '- Hidratacion oral adecuada', 0, 1)
    else:
        pdf.multi_cell(0, 5, 'Recomendaciones especificas para riesgo BAJO:')
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 5, '- Manejo ambulatorio con seguimiento', 0, 1)
        pdf.cell(0, 5, '- Hidratacion oral abundante', 0, 1)
        pdf.cell(0, 5, '- Control de temperatura', 0, 1)
        pdf.cell(0, 5, '- Acudir a control medico si aparecen signos de alarma', 0, 1)
    pdf.ln(3)
    
    # Signos de alarma
    pdf.set_font('Arial', 'B', 10)
    pdf.set_text_color(255, 0, 0)
    pdf.multi_cell(0, 5, 'SIGNOS DE ALARMA (acudir de inmediato a urgencias):')
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 5, '- Dolor abdominal intenso y continuo', 0, 1)
    pdf.cell(0, 5, '- Vomito persistente (mas de 3 veces en 6 horas)', 0, 1)
    pdf.cell(0, 5, '- Sangrado de mucosas (encias, nariz)', 0, 1)
    pdf.cell(0, 5, '- Somnolencia o irritabilidad', 0, 1)
    pdf.cell(0, 5, '- Dificultad para respirar', 0, 1)
    pdf.cell(0, 5, '- Disminucion de la diuresis (orina escasa)', 0, 1)
    pdf.ln(5)
    
    # ============================================================
    # PIE DE PAGINA
    # ============================================================
    pdf.set_font('Arial', 'I', 8)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 5, 'Este informe es una guia de apoyo a la decision clinica.', 0, 1, 'C')
    pdf.cell(0, 5, 'No reemplaza el criterio medico profesional.', 0, 1, 'C')
    pdf.cell(0, 5, 'Ante cualquier duda, consulte con un profesional de la salud.', 0, 1, 'C')
    
    return pdf


def get_datos_historicos():
    """Retorna los datos historicos reales de dengue en la region"""
    
    casos_por_estrato = {
        1: 21585,
        2: 8806,
        3: 2216,
        4: 1385,
        5: 1219,
        6: 1364
    }
    
    total_casos = 36575
    total_graves = 599
    porcentaje_graves = (total_graves / total_casos) * 100
    
    return {
        'total_casos': total_casos,
        'total_graves': total_graves,
        'porcentaje_graves': porcentaje_graves,
        'casos_por_estrato': casos_por_estrato
    }