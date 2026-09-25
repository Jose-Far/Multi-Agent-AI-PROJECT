export const formatIST = (dateString) => {
  if (!dateString) return 'N/A';
  let dStr = typeof dateString === 'string' ? dateString : (dateString.toISOString ? dateString.toISOString() : new Date(dateString).toISOString());
  if (dStr && typeof dStr === 'string' && !dStr.endsWith('Z') && !dStr.includes('+')) {
     dStr += 'Z';
  }
  const date = new Date(dStr);
  if (isNaN(date.getTime())) return 'Invalid Date';
  
  const options = {
    timeZone: 'Asia/Kolkata',
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false
  };
  
  const formatter = new Intl.DateTimeFormat('en-IN', options);
  const parts = formatter.formatToParts(date);
  
  let day = '', month = '', year = '', hour = '', minute = '', second = '';
  parts.forEach(p => {
    if (p.type === 'day') day = p.value;
    if (p.type === 'month') month = p.value;
    if (p.type === 'year') year = p.value;
    if (p.type === 'hour') hour = p.value;
    if (p.type === 'minute') minute = p.value;
    if (p.type === 'second') second = p.value;
  });
  
  return `${day}/${month}/${year} ${hour}:${minute}:${second} IST`;
};

export const formatDuration = (ms) => {
  if (ms === undefined || isNaN(ms)) return null;
  const totalSeconds = Math.floor(ms / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  
  if (hours > 0) {
    return `${hours}h ${minutes}m ${seconds}s`;
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds}s`;
  }
  return `${seconds}s`;
};
