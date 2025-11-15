// app.js — momentos + header shrink + fetch + lógicas de tema
(function(){
  
  // --- LÓGICA DO TEMA (NOVO) ---
  const themeToggle = document.getElementById('themeToggle');
  const body = document.body;

  // Função para aplicar o tema salvo
  function applySavedTheme() {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
      body.classList.add('dark-mode');
    } else {
      body.classList.remove('dark-mode');
    }
  }

  // Listener para o botão de toggle
  themeToggle.addEventListener('click', () => {
    body.classList.toggle('dark-mode');
    // Salva a preferência no localStorage
    if (body.classList.contains('dark-mode')) {
      localStorage.setItem('theme', 'dark');
    } else {
      localStorage.setItem('theme', 'light');
    }
  });
  
  // --- LÓGICA DE NAVEGAÇÃO (Existente) ---
  const btnPratica = document.getElementById('btnPratica');
  const btnVoltar  = document.getElementById('btnVoltar');
  const artigo     = document.getElementById('artigo');
  const pratica    = document.getElementById('pratica');
  const header     = document.getElementById('mainHeader');
  const moment1    = document.getElementById('moment1');
  const moment2    = document.getElementById('moment2');

  // Espaço do header e variável CSS para compensação
  function syncBodyPadding(){
    const h = header.getBoundingClientRect().height;
    body.style.paddingTop = h + 'px';
    document.documentElement.style.setProperty('--header-h', h + 'px');
  }

  // Visibilidade dura
  function setVisible(el, show){
    if (show){
      el.style.display = '';
      el.removeAttribute('hidden');
      el.setAttribute('aria-hidden','false');
      el.removeAttribute('inert');
    } else {
      el.setAttribute('hidden','');
      el.setAttribute('aria-hidden','true');
      el.setAttribute('inert','');
      el.style.display = 'none';
    }
  }

  function activateMoment(which){
    // hard hide both views and inert inactive container
    moment1.classList.remove('is-active');
    moment2.classList.remove('is-active');
    moment1.setAttribute('inert','');
    moment1.setAttribute('aria-hidden','true');
    moment2.setAttribute('inert','');
    moment2.setAttribute('aria-hidden','true');
    setVisible(artigo, false);
    setVisible(pratica, false);

    if (which === 2){
      // Header inicia já recolhido no segundo momento
      header.classList.add('shrink');
      syncBodyPadding();

      moment2.classList.add('is-active');
      moment2.removeAttribute('inert');
      moment2.setAttribute('aria-hidden','false');
      setVisible(pratica, true);
      void pratica.offsetWidth;
      pratica.classList.add('is-showing');
      requestAnimationFrame(() => pratica.classList.remove('is-showing'));
      btnVoltar.hidden = false;
    } else {
      header.classList.remove('shrink');
      syncBodyPadding();

      moment1.classList.add('is-active');
      moment1.removeAttribute('inert');
      moment1.setAttribute('aria-hidden','false');
      setVisible(artigo, true);
      void artigo.offsetWidth;
      artigo.classList.add('is-showing');
      requestAnimationFrame(() => artigo.classList.remove('is-showing'));
      btnVoltar.hidden = true;
    }
  }

  // Inicialização: se entrar com #pratica, recolhe header imediatamente
  if (location.hash === '#pratica') { header.classList.add('shrink'); }
  syncBodyPadding();

  // Eventos de scroll/resize
  window.addEventListener('resize', syncBodyPadding);
  window.addEventListener('scroll', () => {
    // Se estiver no momento 1, controlar shrink por rolagem; no momento 2 permanece shrink
    if (moment1.classList.contains('is-active')) {
      if (window.scrollY > 60) header.classList.add('shrink');
      else header.classList.remove('shrink');
      syncBodyPadding();
    }
  });


  // --- Lógica para enviar o formulário de TREINO ---

  async function handleTrainSubmit(event) {
    event.preventDefault(); // Impede o recarregamento da página
    
    const resultsDiv = document.getElementById('trainResults');
    const btn = document.getElementById('btnTrainModel');
    
    // 1. Mostrar loading
    resultsDiv.innerHTML = '<p>Treinando o modelo... Isso pode levar um momento.</p>';
    btn.disabled = true;
    btn.textContent = 'Treinando...';

    // 2. Coletar os dados do formulário
   const trainParams = {
      test_size: Number(document.getElementById('param_test_size').value),
      activation_function: document.getElementById('param_activation').value,
      optimiser: document.getElementById('param_optimiser').value,
      loss_function: document.getElementById('param_loss').value,
      epochs: Number(document.getElementById('param_epochs').value),
      batch_size: Number(document.getElementById('param_batch_size').value),
      num_hidden_layers: Number(document.getElementById('param_num_hidden_layers').value),
      neurons_per_layer: Number(document.getElementById('param_neurons_per_layer').value),
      dropout_rate: Number(document.getElementById('param_dropout_rate').value)
    };

    // 3. Enviar para o backend (Flask)
    try {
      const response = await fetch('http://127.0.0.1:5000/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(trainParams),
      });

      if (!response.ok) {
        throw new Error(`Erro na rede: ${response.statusText}`);
      }

      const results = await response.json();

      // 4. Mostrar os resultados
      resultsDiv.innerHTML = `
        <p style="color: var(--success-text);">${results.message}</p>
        <hr class="separator" style="margin: 12px 0;">
        <p><strong>Acurácia na Validação:</strong> ${(results.val_accuracy * 100).toFixed(2)}%</p>
        <p><strong>Perda (Loss) na Validação:</strong> ${results.val_loss.toFixed(4)}</p>
      `;

    } catch (error) {
      // 5. Mostrar erro
      console.error('Erro ao treinar:', error);
      resultsDiv.innerHTML = `<p style="color: var(--error-text);"><b>Falha no Treinamento:</b><br>${error.message}. Verifique o console do navegador e o terminal do Flask.</p>`;
    } finally {
      // 6. Reativar o botão
      btn.disabled = false;
      btn.textContent = 'Treinar Modelo';
    }
  }

  // --- Lógica para enviar o formulário de PREVISÃO ---

  async function handlePredictSubmit(event) {
    event.preventDefault(); // Impede o recarregamento da página
    
    const resultsDiv = document.getElementById('predictResults');
    const btn = document.getElementById('btnPredict');
    
    // 1. Mostrar loading
    resultsDiv.innerHTML = '<p>Calculando previsão...</p>';
    btn.disabled = true;
    btn.textContent = 'Calculando...';

    // 2. Coletar os dados do formulário
    const predictionData = {
      // O backend espera os nomes exatos das colunas
      Pclass: Number(document.getElementById('pred_pclass').value),
      Sex: document.getElementById('pred_sex').value,
      Age: Number(document.getElementById('pred_age').value),
      SibSp: Number(document.getElementById('pred_sibsp').value),
      Parch: Number(document.getElementById('pred_parch').value),
      Fare: Number(document.getElementById('pred_fare').value),
      Embarked: document.getElementById('pred_embarked').value
    };

    // 3. Enviar para o backend (Flask)
    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(predictionData),
      });

      if (!response.ok) {
        // Tenta ler a mensagem de erro específica do backend
        const errorData = await response.json().catch(() => ({})); 
        throw new Error(errorData.error || `Erro na rede: ${response.statusText}`);
      }

      const results = await response.json();

      // 4. Mostrar os resultados
      const probability = (results.probabilidade_sobreviver * 100).toFixed(2);
      const message = results.mensagem;
      const survived = results.previsao === 1; // Booleano para facilitar

      // Define classes e ícones com base no resultado
      const messageClass = survived ? 'prediction-survived' : 'prediction-not-survived';
      const iconClass = survived ? 'check_circle' : 'cancel'; // Ícones do Material Symbols

      resultsDiv.innerHTML = `
        <div class="prediction-header ${messageClass}">
          <span class="material-symbols-outlined prediction-icon">${iconClass}</span>
          <span class="prediction-text">${message}</span>
        </div>
        <div class="prediction-details">
          <p>Probabilidade de Sobreviver: <strong>${probability}%</strong></p>
        </div>
      `;

    } catch (error) {
      // 5. Mostrar erro
      console.error('Erro ao prever:', error);
      resultsDiv.innerHTML = `<p style="color: var(--error-text);"><b>Falha na Previsão:</b><br>${error.message}. Verifique o console e o terminal do Flask.</p>`;
    
    } finally {
      // 6. Reativar o botão
      btn.disabled = false;
      btn.textContent = 'Prever Sobrevivência';
    }
  }
  
  // Navegação por hash
  function applyMomentFromHash(){
    const inPratica = (location.hash === '#pratica');
    if (inPratica){ 
      activateMoment(2); 
    } else { 
      activateMoment(1); 
    }
  }

  // Registra os listeners do formulário quando o DOM estiver pronto
  window.addEventListener('DOMContentLoaded', () => {
    // Aplica o tema salvo IMEDIATAMENTE
    applySavedTheme();

    applyMomentFromHash();
    
    // Listener do formulário de TREINO
    const trainForm = document.getElementById('trainForm');
    if (trainForm) {
      trainForm.addEventListener('submit', handleTrainSubmit);
    }
    
    // Listener do formulário de PREVISÃO
    const predictForm = document.getElementById('predictForm');
    if (predictForm) {
      predictForm.addEventListener('submit', handlePredictSubmit);
    }
  });

  // Listeners de navegação
  window.addEventListener('load', applyMomentFromHash);
  window.addEventListener('pageshow', applyMomentFromHash);
  window.addEventListener('popstate', applyMomentFromHash);

  // Botões
  btnPratica.addEventListener('click', () => {
    activateMoment(2);
    syncBodyPadding();
    pratica.scrollIntoView({ behavior: 'smooth', block: 'start' });
    if (location.hash !== '#pratica') history.replaceState({}, '', '#pratica');
  });

  btnVoltar.addEventListener('click', () => {
    activateMoment(1);
    syncBodyPadding();
    artigo.scrollIntoView({ behavior: 'smooth', block: 'start' });
    if (location.hash) history.replaceState({}, '', ' ');
  });
})();
